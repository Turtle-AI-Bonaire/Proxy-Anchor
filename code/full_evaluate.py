import torch, math, time, argparse, json, os, sys
import random, dataset, utils, losses, net
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from embedding_averager import EmbeddingAverager
# Make sure PIL is available:
from PIL import Image, ImageDraw, ImageFont

from dataset.BonaireTurtlesDataset import BonaireTurtlesDataset
from utils import save_r4_to_txt

# ResNet Models
from net.resnet import Resnet18, Resnet50, Resnet101

# Dataset mapping for 'bon' only
dataset_map = {
    'bon': BonaireTurtlesDataset
}

transform = T.Compose([
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

# Initialize random seed for reproducibility
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Argument parser
parser = argparse.ArgumentParser(description='Single-Image Retrieval Mode using ResNet models')
parser.add_argument('--dataset', default='bon', help='Training dataset, e.g. bon')
parser.add_argument('--embedding-size', default=512, type=int, dest='sz_embedding', help='Size of embedding that is appended to backbone model.')
parser.add_argument('--batch-size', default=50, type=int, dest='sz_batch', help='Number of samples per batch.')
parser.add_argument('--gpu-id', default=0, type=int, help='ID of GPU that is used for training.')
parser.add_argument('--workers', default=4, type=int, dest='nb_workers', help='Number of workers for dataloader.')
parser.add_argument('--model', default='resnet18', help='Model for training (resnet18, resnet50, resnet101)')
parser.add_argument('--l2-norm', default=1, type=int, help='L2 normalization')
parser.add_argument('--resume', default='', help='Path of resuming model')
parser.add_argument('--remark', default='', help='Any remark')
parser.add_argument('--top-n', type=int, default=5, help="Number of top-similar images to display alongside the query.")

args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Data Root Directory
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ---------- LOAD THE 'bon' DATASET ----------
dataset_class = dataset_map['bon']
ev_dataset = dataset_class(
    root=data_root,
    mode='val',
    ignoreThreshold=0,
    transform=dataset.utils.make_transform(
        is_train=False,
        is_inception=False  # Ensure we aren't using Inception-specific transformations
    )
)

dl_ev = torch.utils.data.DataLoader(
    ev_dataset,
    batch_size=args.sz_batch,
    shuffle=False,
    num_workers=args.nb_workers,
    pin_memory=True
)
print(f"Loaded 'bon' dataset for retrieval (N={len(ev_dataset)})")

# ---------- LOAD THE RESNET BACKBONE MODEL ----------
if args.model == 'resnet18':
    model = Resnet18(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=1)
elif args.model == 'resnet50':
    model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=1)
elif args.model == 'resnet101':
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

averager = EmbeddingAverager(reduction='mean').cuda()

# 1) Initialize empty lists to store embeddings, paths, and labels
all_embeddings = []
all_paths = []
all_labels = []

# 2) Process dataset in batches and compute embeddings
with torch.no_grad():
    for batch_idx, (img_tensors, lbls) in tqdm(enumerate(dl_ev), desc="Computing all embeddings", total=len(dl_ev)):
        # Move batch to GPU
        img_tensors = img_tensors.cuda()

        # Get embeddings for the current batch
        batch_embeddings = model(img_tensors)  # shape: (batch_size, embedding_dim)
        batch_embeddings = batch_embeddings.cpu().numpy()

        # Append batch embeddings and corresponding data
        all_embeddings.append(batch_embeddings)
        all_labels.append(lbls)

        # Retrieve image paths (make sure we don't go out of bounds)
        batch_start = batch_idx * args.sz_batch
        batch_end = min((batch_idx + 1) * args.sz_batch, len(ev_dataset.im_paths))  # Avoid going beyond the dataset size
        batch_paths = ev_dataset.im_paths[batch_start:batch_end]  # Slice the list properly
        all_paths.extend(batch_paths)
        
em_labels = torch.cat(all_labels)
all_embeddings = np.vstack(all_embeddings)  # shape: (N_dataset, embed_dim)
print(all_embeddings.shape)
all_embedding_anchors = averager(all_embeddings, labels=em_labels) 
print(all_embedding_anchors.shape)
# 3) Retrieve query image embedding
with torch.no_grad():
    query_embeddings = []

    for image_path in query_image_paths:
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0).cuda()
        emb = model(img).cpu()
        query_embeddings.append(emb)

    query_embeddings = torch.cat(query_embeddings, dim=0)

    query_emb, _, _ = averager(query_embeddings, labels=torch.zeros(query_embeddings.size(0)).long())  # Dummy labels
    query_emb = query_emb.squeeze()
    print(f"Query image embedding shape: {query_emb.shape}")

# 4) Compute cosine similarity between query embedding and all embeddings
sims = np.dot(all_embeddings, query_emb)

# 5) Sort by descending similarity. Exclude the query itself at index q_idx.
sorted_indices = np.argsort(-sims)
top_n = args.top_n
topk_indices = sorted_indices[:top_n]

topk_paths = [all_paths[i] for i in topk_indices]
topk_sims  = [sims[i] for i in topk_indices]
topk_labels = [all_labels[i] for i in topk_indices]

print(f"Top {top_n} matches:")
for rank, (p, sim, lbl) in enumerate(zip(topk_paths, topk_sims, topk_labels), start=1):
    print(f"  {rank}. {os.path.basename(p)} (label={lbl}, sim={sim:.4f})")

# 6) Compose a single “montage” PNG that shows those Top-N images side by side.
cell_w, cell_h_img = 128, 128
text_h = 20
cell_h = cell_h_img + text_h

montage_w = cell_w * top_n
montage_h = cell_h

montage = Image.new('RGB', (montage_w, montage_h), color=(255, 255, 255))
draw = ImageDraw.Draw(montage)
font = ImageFont.load_default()

for i, idx in enumerate(topk_indices):
    img_path = all_paths[idx]
    with Image.open(img_path) as im:
        im_disp = im.convert('RGB').resize((cell_w, cell_h_img), Image.BILINEAR)
    x0 = i * cell_w
    montage.paste(im_disp, (x0, 0))

    # Draw filename (basename) under each patch
    basename = os.path.basename(img_path)
    text = basename
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h_actual = bbox[2] - bbox[0], bbox[3] - bbox[1]
    text_x = x0 + (cell_w - text_w)//2
    text_y = cell_h_img + ( (text_h - text_h_actual)//2 )
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

filename = os.path.splitext(os.path.basename(query_image_paths[0]))[0]

out_name = f"bon_query_{filename}_top{top_n}.png"
out_path = os.path.join("query_results", out_name)
montage.save(out_path)
print(f"Saved retrieval montage to '{out_name}'")
sys.exit(0)
