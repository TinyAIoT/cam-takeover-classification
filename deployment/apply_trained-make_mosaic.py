#!/usr/bin/env python3
import argparse
import os
import re
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import torchvision

# ------------------------------------------------------------
# Filename parsing (same as your convention)
# ------------------------------------------------------------
RIDE_RE = re.compile(r'ride_(\d+)', re.IGNORECASE)
OFFSET_RE = re.compile(r'offset(\d+)', re.IGNORECASE)
TAKEOVER_EXPLICIT_RE = re.compile(r'takeover[-_](\d+)', re.IGNORECASE)
# Fallback: number immediately before "_offset"
NUM_BEFORE_OFFSET_RE = re.compile(r'[_-](\d+)_offset', re.IGNORECASE)

def parse_from_name(fname):
    """
    Extract (ride, takeover, offset) from flexible filenames like:
      - ride_2_mostly_takeover_takeover_1_offset012.png
      - ride_1_mostly_not_takeover_cyclist-takeover-2_offset019.png
      - ride_1_mostly_not_takeover_other_2_offset177.png
      - ride_2_mostly_not_takeover_other_offset011.png  (no takeover -> 0)
      - ride_3_mostly_not_takeover_cyclist_takeover_1_offset001.png
    Returns (ride:int, takeover:int, offset:int) or None if required parts missing.
    """
    base = os.path.basename(fname)

    m_ride = RIDE_RE.search(base)
    m_offset = OFFSET_RE.search(base)
    if not m_ride or not m_offset:
        return None
    ride = int(m_ride.group(1))
    offset = int(m_offset.group(1))

    m_take = TAKEOVER_EXPLICIT_RE.search(base)
    if m_take:
        takeover = int(m_take.group(1))
    else:
        m_num_before = NUM_BEFORE_OFFSET_RE.search(base)
        takeover = int(m_num_before.group(1)) if m_num_before else 0  # default when absent

    return ride, takeover, offset

# ------------------------------------------------------------
# Data loading for inference
# ------------------------------------------------------------
class ImageRecord:
    __slots__ = ("path", "ride", "takeover", "offset", "pil")
    def __init__(self, path, ride, takeover, offset, pil):
        self.path = path
        self.ride = ride
        self.takeover = takeover
        self.offset = offset
        self.pil = pil

def scan_folder(folder):
    recs = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"Skipping non-image file: {fname}")
            continue
        parsed = parse_from_name(fname)
        if not parsed:
            # skip files that don't match the grouping pattern
            print(f"Skipping unrecognized filename: {fname}")
            continue
        ride, takeover, offset = parsed
        path = os.path.join(folder, fname)
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error opening {fname}: {e}")
            continue
        recs.append(ImageRecord(path, ride, takeover, offset, pil))
    return recs

# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------
def load_model(model_path, device, num_classes=2):
    """
    Load SqueezeNet1_1 with custom classifier head and state_dict weights.
    """
    # Build base SqueezeNet
    # model = torchvision.models.squeezenet1_1(weights="DEFAULT")
    # model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    
    # Load your state_dict
    # state_dict = torch.load(model_path, map_location=device)
    # model.load_state_dict(state_dict)
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model.to(device)

# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------
@torch.no_grad()
def run_inference(records, model, device, mean, std, input_size=None, batch_size=64, class_names=("not_takeover", "takeover")):
    """
    Returns dict: path -> (pred_class_name, pred_prob)
    """
    # Build transform pipeline (train-time transforms)
    tfms = []
    if input_size is not None:
        tfms.append(transforms.Resize((input_size, input_size), interpolation=Image.BILINEAR))
    tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    tfm = transforms.Compose(tfms)

    # Prepare tensors in batches
    preds = {}
    # Simple batching without DataLoader for clarity
    def batch(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i+n]

    for chunk in batch(records, batch_size):
        batch_imgs = []
        batch_paths = []
        for rec in chunk:
            x = tfm(rec.pil)
            batch_imgs.append(x)
            batch_paths.append(rec.path)
        x = torch.stack(batch_imgs, dim=0).to(device)

        logits = model(x)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        probs = F.softmax(logits, dim=1)

        confs, idxs = probs.max(dim=1)
        for pth, idx, conf in zip(batch_paths, idxs.tolist(), confs.tolist()):
            # Map class index to name; default assumes 0=not_takeover, 1=takeover
            name = class_names[idx] if 0 <= idx < len(class_names) else f"class_{idx}"
            preds[pth] = (name, float(conf))
    return preds

num_not_takeover = 0
num_takeover = 0
# ------------------------------------------------------------
# Mosaic building (with prediction labels above)
# ------------------------------------------------------------
def build_grid_from_predictions(records, predictions,
                                max_cols=25, gap=4, margin=20, border=2,
                                text_height=16, bg=(255, 255, 255)):
    """
    records: list[ImageRecord]
    predictions: dict path -> (classname, prob)
    Group by (ride, takeover). Sort each group by offset. Draw text+border using predicted class.
    """
    global num_not_takeover, num_takeover
    # Ensure consistent tile size
    first_w, first_h = records[0].pil.size
    for r in records:
        if r.pil.size != (first_w, first_h):
            r.pil = r.pil.resize((first_w, first_h), Image.LANCZOS)

    # Group
    groups = defaultdict(list)
    for r in records:
        if r.path not in predictions:
            continue
        cname, prob = predictions[r.path]
        groups[(r.ride, r.takeover)].append({
            "img": r.pil,
            "offset": r.offset,
            "classname": cname,
            "prob": prob
        })
        if cname == "takeover":
            num_takeover += 1
        else:
            num_not_takeover += 1
    # Sort by offset
    for k in groups:
        groups[k].sort(key=lambda d: d["offset"])

    if not groups:
        raise ValueError("No groups or predictions to render.")

    w, h = first_w, first_h
    ordered_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1]))
    rows = len(ordered_keys)
    cols = min(max_cols, max(len(groups[k]) for k in ordered_keys))

    cell_w = w + 2*border
    cell_h = h + 2*border + text_height

    canvas_w = margin*2 + (cols * cell_w) + (cols-1)*gap if cols > 0 else margin*2
    canvas_h = margin*2 + (rows * cell_h) + (rows-1)*gap if rows > 0 else margin*2

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    green = (0, 170, 0)
    red = (220, 30, 30)

    for r, key in enumerate(ordered_keys):
        row_items = groups[key][:max_cols]  # enforce max width
        for c, item in enumerate(row_items):
            x0 = margin + c * (cell_w + gap)
            y0 = margin + r * (cell_h + gap)

            outline_color = green if item["classname"] == "takeover" else red
            text = f"{item['classname']} {item['prob']:.4f}"

            # text size (Pillow >=9.2): use textbbox
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]

            text_x = x0 + (cell_w - text_w) // 2
            text_y = y0
            draw.text((text_x, text_y), text, fill=outline_color, font=font)

            # image position (below text space)
            img_y = y0 + text_height
            canvas.paste(item["img"], (x0 + border, img_y + border))

            # border around image area
            for t in range(border):
                draw.rectangle(
                    [x0 + t, img_y + t, x0 + cell_w - 1 - t, img_y + cell_h - text_height - 1 - t],
                    outline=outline_color
                )

    return canvas

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def parse_float_list(arg):
    # e.g. "0.485,0.456,0.406"
    vals = [float(x.strip()) for x in arg.split(",")]
    return vals

def get_default_device():
    """Pick GPU if available, else CPU
    Returns:
        torch.device: Device to use for training (CPU, CUDA, or MPS)
    """
    if torch.cuda.is_available():
        print(f"Using CUDA for training.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print(f"Using MPS for training.")
        return torch.device("mps")
    else:
        print(f"Using CPU for training.")
        return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description="Classify tiles with a PyTorch .pt model and build a labeled grid mosaic.")
    parser.add_argument("--input_folder", help="Folder with JPEG tiles.")
    parser.add_argument("--model_path", help="Path to .pt model (TorchScript or full pickled model).")
    parser.add_argument("-o", "--output", default="mosaic_pred.jpg", help="Output image path.")
    parser.add_argument("--mean", type=parse_float_list, required=True, help="Normalize mean, comma-separated (e.g., 0.485,0.456,0.406).")
    parser.add_argument("--std", type=parse_float_list, required=True, help="Normalize std, comma-separated (e.g., 0.229,0.224,0.225).")
    parser.add_argument("--input-size", type=int, default=None, help="Optional resize to NxN before ToTensor/Normalize (default: no resize).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--classes", type=str, default="not_takeover,takeover", help="Comma-separated class names in model index order.")
    parser.add_argument("--max-cols", type=int, default=25, help="Max tiles per row.")
    parser.add_argument("--gap", type=int, default=4, help="Gap between tiles (px).")
    parser.add_argument("--margin", type=int, default=20, help="Outer margin (px).")
    parser.add_argument("--border", type=int, default=2, help="Border thickness (px).")
    parser.add_argument("--text-height", type=int, default=16, help="Reserved text height per tile (px).")
    args = parser.parse_args()

    device = get_default_device()
    class_names = tuple([s.strip() for s in args.classes.split(",") if s.strip()])

    records = scan_folder(args.input_folder)
    if not records:
        raise SystemExit("No matching images found in the folder (or filenames didn't match the expected pattern).")

    model = load_model(args.model_path, device=device)

    preds = run_inference(
        records=records,
        model=model,
        device=device,
        mean=args.mean,
        std=args.std,
        input_size=args.input_size,
        batch_size=args.batch_size,
        class_names=class_names
    )

    mosaic = build_grid_from_predictions(
        records=records,
        predictions=preds,
        max_cols=args.max_cols,
        gap=args.gap,
        margin=args.margin,
        border=args.border,
        text_height=args.text_height,
        bg=(255, 255, 255)
    )

    mosaic.save(args.output)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
    print(f"Number of 'takeover' images: {num_takeover}")
    print(f"Number of 'not_takeover' images: {num_not_takeover}")
