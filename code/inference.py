import torch, math, time, argparse, json, os, sys
import random, dataset, utils, losses, net
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from embedding_averager import EmbeddingAverager
from PIL import Image, ImageDraw, ImageFont

from dataset.BonaireTurtlesDataset import BonaireTurtlesDataset
from utils import save_r4_to_txt

from net.resnet import Resnet18, Resnet50, Resnet101

dataset_map = {
    'bon': BonaireTurtlesDataset
}

transform = T.Compose([
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

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
parser.add_argument('--average-embeddings', type=bool, default=False, help="Whether to average the database embeddings before comparison.")

args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

dataset_class = dataset_map['bon']
ev_dataset = dataset_class(
    root=data_root,
    mode='full',
    ignoreThreshold=0,
    transform=dataset.utils.make_transform(is_train=False, is_inception=False)
)

dl_ev = torch.utils.data.DataLoader(
    ev_dataset,
    batch_size=args.sz_batch,
    shuffle=False,
    num_workers=args.nb_workers,
    pin_memory=True
)
print(f"Loaded 'bon' dataset for retrieval (N={len(ev_dataset)})")

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

# ------ Load Image Paths --------
def get_image_paths(folder_path="query_images"):
    if not os.path.exists(folder_path):
        print("query_images folder does not exist.")
        sys.exit(0)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    if len(image_paths) == 0:
        print("No image files found in the query_images folder.")
        sys.exit(0)
    return image_paths

query_image_paths = get_image_paths()
transform = dataset.utils.make_transform(is_train=False, is_inception=False)

averager = EmbeddingAverager(reduction='mean').cuda()

# 1) Initialize empty lists for embeddings, paths, and labels
all_embeddings = []
all_paths = []
all_labels = []

# 2) Process dataset in batches and compute embeddings
with torch.no_grad():
    for batch_idx, (img_tensors, lbls) in tqdm(enumerate(dl_ev), desc="Computing all embeddings", total=len(dl_ev)):
        img_tensors = img_tensors.cuda()
        batch_embeddings = model(img_tensors)
        batch_embeddings = batch_embeddings.cpu().numpy()

        all_embeddings.append(batch_embeddings)
        all_labels.extend(lbls.numpy())

        batch_start = batch_idx * args.sz_batch
        batch_end = min((batch_idx + 1) * args.sz_batch, len(ev_dataset.im_paths))
        batch_paths = ev_dataset.im_paths[batch_start:batch_end]
        all_paths.extend(batch_paths)

all_embeddings = np.vstack(all_embeddings)

# 3) Retrieve query image embedding
with torch.no_grad():
    query_embeddings = []
    for image_path in query_image_paths:
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0).cuda()
        emb = model(img).cpu()
        query_embeddings.append(emb)

    query_embeddings = torch.cat(query_embeddings, dim=0)

    query_emb, _, _ = averager(query_embeddings, labels=torch.zeros(query_embeddings.size(0)).long())
    query_emb = query_emb.squeeze()
    print(f"Query image embedding shape: {query_emb.shape}")

# 4) Conditionally compute embeddings for each identity and average them if specified
identity_embeddings = {}
if args.average_embeddings:
    for idx, lbl in enumerate(all_labels):
        if lbl not in identity_embeddings:
            identity_embeddings[lbl] = []
        identity_embeddings[lbl].append(all_embeddings[idx])

    # Average embeddings for each identity
    identity_avg_embeddings = {}
    for lbl, embeddings in identity_embeddings.items():
        identity_avg_embeddings[lbl] = np.mean(embeddings, axis=0)

    # Use averaged embeddings for similarity calculation
    sims = {lbl: np.dot(query_emb, emb) for lbl, emb in identity_avg_embeddings.items()}
else:
    # Use individual embeddings for each sample
    sims = {lbl: np.dot(query_emb, emb) for lbl, emb in zip(all_labels, all_embeddings)}

# 5) Sort by descending similarity
sorted_labels = sorted(sims.items(), key=lambda x: x[1], reverse=True)
top_n = args.top_n
topk_labels = sorted_labels[:top_n]

topk_paths = []
topk_sims = []

for lbl, _ in topk_labels:
    idxs = [i for i, label in enumerate(all_labels) if label == lbl]
    topk_paths.append(all_paths[idxs[0]])  # Choose the first image path from each identity
    topk_sims.append(sims[lbl])

# 6) Display top matches
print(f"Top {top_n} matches:")
for rank, (p, sim) in enumerate(zip(topk_paths, topk_sims), start=1):
    print(f"  {rank}. {os.path.basename(p)} (sim={sim:.4f})")

# 7) Create montage
cell_w, cell_h_img = 128, 128
text_h = 20
cell_h = cell_h_img + text_h

montage_w = cell_w * top_n
montage_h = cell_h

montage = Image.new('RGB', (montage_w, montage_h), color=(255, 255, 255))
draw = ImageDraw.Draw(montage)
font = ImageFont.load_default()

for i, idx in enumerate(topk_paths):
    img_path = idx
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
