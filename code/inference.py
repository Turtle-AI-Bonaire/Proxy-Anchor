import torch, math, time, argparse, json, os, sys
import random, dataset, utils, losses, net
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T

from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate

# Make sure PIL is available:
from PIL import Image, ImageDraw, ImageFont

from dataset.ComboDataset import CombinedTurtlesDataset
from dataset.Inshop import Inshop_Dataset
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *
from dataset.BonaireTurtlesDataset import BonaireTurtlesDataset
from dataset.SeaTurtleIDHeadsDataset import SeaTurtleIDHeadsDataset
from dataset.AmvrakikosDataset import AmvrakikosDataset
from utils import save_r4_to_txt

dataset_map = {
    'tih': SeaTurtleIDHeadsDataset,
    'amv': AmvrakikosDataset,
    'combo': CombinedTurtlesDataset,
    'bon': BonaireTurtlesDataset
}

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser(description=
    'Official implementation of `Proxy Anchor Loss for Deep Metric Learning`'  
    + 'Our code is modified from `https://github.com/dichotomies/proxy-nca`'
)
parser.add_argument('--dataset', 
    default='cub',
    help = 'Training dataset, e.g. cub, cars, SOP, Inshop'
)
parser.add_argument('--embedding-size', default = 512, type = int,
    dest = 'sz_embedding',
    help = 'Size of embedding that is appended to backbone model.'
)
parser.add_argument('--batch-size', default = 150, type = int,
    dest = 'sz_batch',
    help = 'Number of samples per batch.'
)
parser.add_argument('--gpu-id', default = 0, type = int,
    help = 'ID of GPU that is used for training.'
)
parser.add_argument('--workers', default = 4, type = int,
    dest = 'nb_workers',
    help = 'Number of workers for dataloader.'
)
parser.add_argument('--model', default = 'bn_inception',
    help = 'Model for training'
)
parser.add_argument('--l2-norm', default = 1, type = int,
    help = 'L2 normlization'
)
parser.add_argument('--resume', default = '',
    help = 'Path of resuming model'
)
parser.add_argument('--remark', default = '',
    help = 'Any remark'
)

# << NEW ARGS for single-image retrieval mode >>
parser.add_argument('--query-index', type=int, default=None,
    help="Index of one image in the 'bon' dataset to use as the query. If provided, the script will do single-image recall instead of global evaluation.")
parser.add_argument('--top-n', type=int, default=5,
    help="Number of top-similar images to display alongside the query in retrieval mode (only valid when --query-index is set).")

args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Data Root Directory
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ---------- BUILD DATASET (only 'bon' needs special handling for retrieval) ----------
if args.dataset in dataset_map:
    dataset_class = dataset_map[args.dataset]
    # Use the “eval” split for everything, including retrieval
    ev_dataset = dataset_class(
        root=data_root,
        mode='eval',
        transform=dataset.utils.make_transform(
            is_train=True,
            is_inception=(args.model == 'bn_inception')
        )
    )
    # For 'bon', we explicitly re-instantiate to override ignoreThreshold=0
    if args.dataset == 'bon':
        ev_dataset = BonaireTurtlesDataset(
            root=data_root,
            mode='eval',
            ignoreThreshold=0,
            transform=dataset.utils.make_transform(
                is_train=True,
                is_inception=(args.model == 'bn_inception')
            )
        )
    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )
    print("Loaded '%s' validation/retrieval dataset (N=%d)" % (args.dataset, len(ev_dataset)))

elif args.dataset == 'Inshop':
    print("Correct Val Dataset: Inshop (handled separately)")
    query_dataset = Inshop_Dataset(
        root=data_root,
        mode='query',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        )
    )
    dl_query = torch.utils.data.DataLoader(
        query_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )
    gallery_dataset = Inshop_Dataset(
        root=data_root,
        mode='gallery',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        )
    )
    dl_gallery = torch.utils.data.DataLoader(
        gallery_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

else:
    # For other datasets not in dataset_map (and not Inshop)
    ev_dataset = dataset.load(
        name=args.dataset,
        root=data_root,
        mode='eval',
        transform=dataset.utils.make_transform(
            is_train=False,
            is_inception=(args.model == 'bn_inception')
        )
    )
    dl_ev = torch.utils.data.DataLoader(
        ev_dataset,
        batch_size=args.sz_batch,
        shuffle=False,
        num_workers=args.nb_workers,
        pin_memory=True
    )

# ---------- BUILD THE BACKBONE MODEL ----------
if args.model.find('googlenet')+1:
    model = googlenet(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=1)
elif args.model.find('bn_inception')+1:
    model = bn_inception(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=1)
elif args.model.find('resnet18')+1:
    model = Resnet18(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=1)
elif args.model.find('resnet50')+1:
    model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=1)
elif args.model.find('resnet101')+1:
    model = Resnet101(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=1)
else:
    raise ValueError(f"Unknown model: {args.model}")

model = model.cuda()
if args.gpu_id == -1:
    model = torch.nn.DataParallel(model)

if os.path.isfile(args.resume):
    print(f"=> loading checkpoint {args.resume}")
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint)
else:
    print(f"=> No checkpoint found at {args.resume}")
    sys.exit(0)

model.eval()


# ---------------------- SINGLE-IMAGE RETRIEVAL MODE (for ‘bon’ only) ----------------------
if args.dataset == 'bon' and args.query_index is not None:
    # 1) Build embeddings for every image in the 'bon' dataset.
    #    We will iterate over ev_dataset directly so that we can also recover each original file path.
    #
    #    We assume that ev_dataset.imgs is a list of (path, label) tuples.
    #    If your dataset class uses a different attribute to store paths,
    #    replace ev_dataset.imgs[idx][0] with the correct path accessor.
    #
    all_embeddings = []
    all_paths = []
    all_labels = []

    with torch.no_grad():
        for idx in tqdm(range(len(ev_dataset)), desc="Computing all embeddings"):
            # __getitem__ returns (transformed_image_tensor, label) by default.
            # If your BonaireTurtlesDataset __getitem__ already returns path, adapt as needed.
            img_tensor, lbl = ev_dataset[idx]
            # embed returns a 1×D tensor
            emb = model(img_tensor.unsqueeze(0).cuda())  # (1 × embed_dim)
            emb = emb.cpu().numpy().copy().reshape(-1)    # (embed_dim,)
            all_embeddings.append(emb)

            # get the original file path (assuming ev_dataset.imgs exists)
            #  – adjust if your dataset stores file‐list under a different attribute
            path = ev_dataset.im_paths[idx]
            all_paths.append(path)
            all_labels.append(lbl)

    all_embeddings = np.stack(all_embeddings, axis=0)  # shape: (N_dataset, embed_dim)

    # 2) Take the single “query” image at query_index, get its embedding:
    q_idx = args.query_index
    if not (0 <= q_idx < len(ev_dataset)):
        print(f"Error: query-index {q_idx} is out of range for dataset of size {len(ev_dataset)}.")
        sys.exit(1)
    query_emb = all_embeddings[q_idx]

    # 3) Compute cosine similarity between query embedding and all embeddings
    #    If your model output is already L2‐normalized (args.l2_norm=1), we can just do a dot product.
    #    Otherwise, uncomment the L2‐normalization lines below.
    #
    # # Optionally L2‐normalize every embedding row:
    # eps = 1e-10
    # norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True) + eps
    # all_embeddings_normed = all_embeddings / norms
    # query_emb_normed = query_emb / (np.linalg.norm(query_emb) + eps)
    # sims = np.dot(all_embeddings_normed, query_emb_normed)
    #
    sims = np.dot(all_embeddings, query_emb)

    # 4) Sort by descending similarity. Exclude the query itself at index q_idx.
    sorted_indices = np.argsort(-sims)
    # Remove query itself
    sorted_indices = [i for i in sorted_indices if i != q_idx]
    top_n = args.top_n
    topk_indices = sorted_indices[:top_n]

    topk_paths = [all_paths[i] for i in topk_indices]
    topk_sims  = [sims[i] for i in topk_indices]
    topk_labels = [all_labels[i] for i in topk_indices]

    print(f"Query image: {all_paths[q_idx]}")
    print("Top %d matches:" % top_n)
    for rank, (p, sim, lbl) in enumerate(zip(topk_paths, topk_sims, topk_labels), start=1):
        print(f"  {rank}. {os.path.basename(p)} (label={lbl}, sim={sim:.4f})")

    # 5) Compose a single “montage” PNG that shows those Top‐N images side by side.
    #    We’ll take each image, resize to 128×128, and put 20px of white space below for text.
    #
    cell_w, cell_h_img = 128, 128
    text_h = 20
    cell_h = cell_h_img + text_h

    montage_w = cell_w * top_n
    montage_h = cell_h

    montage = Image.new('RGB', (montage_w, montage_h), color=(255,255,255))
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()

    for i, idx in enumerate(topk_indices):
        img_path = all_paths[idx]
        # Open original image, resize to (128×128)
        with Image.open(img_path) as im:
            im_disp = im.convert('RGB').resize((cell_w, cell_h_img), Image.BILINEAR)
        # Paste into montage
        x0 = i * cell_w
        montage.paste(im_disp, (x0, 0))

        # Draw filename (basename) or label under each patch
        basename = os.path.basename(img_path)
        text = basename
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h_actual = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_x = x0 + (cell_w - text_w)//2
        text_y = cell_h_img + ( (text_h - text_h_actual)//2 )
        draw.text((text_x, text_y), text, fill=(0,0,0), font=font)

    out_name = f"bon_query_{q_idx}_top{top_n}.png"
    montage.save(out_name)
    print(f"Saved retrieval montage to '{out_name}'")
    sys.exit(0)


# ---------------------- NORMAL EVALUATION MODE ----------------------
with torch.no_grad():
    print("**Evaluating...**")
    if args.dataset == 'Inshop':
        Recalls = utils.evaluate_cos_Inshop(model, dl_query, dl_gallery)

    elif args.dataset != 'SOP':
        Recalls = utils.evaluate_cos(model, dl_ev)
        save_path = os.path.join(os.path.dirname(__file__), "logs.txt")
        save_r4_to_txt(Recalls, save_path)
    else:
        Recalls = utils.evaluate_cos_SOP(model, dl_ev)