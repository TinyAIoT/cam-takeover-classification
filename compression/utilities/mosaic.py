from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import os
import re

# Regex patterns from apply_trained-make_mosaic.py
RIDE_RE = re.compile(r'ride_(\d+)', re.IGNORECASE)
OFFSET_RE = re.compile(r'offset(\d+)', re.IGNORECASE)
TAKEOVER_EXPLICIT_RE = re.compile(r'takeover[-_](\d+)', re.IGNORECASE)
NUM_BEFORE_OFFSET_RE = re.compile(r'[_-](\d+)_offset', re.IGNORECASE)

def parse_from_name(fname):

    base = os.path.basename(fname)
    # Get the class folder from the second-to-last component in the path
    parts = os.path.normpath(fname).split(os.sep)
    if len(parts) >= 2:
        folder = parts[-2]
    else:
        folder = ""
    if folder.lower() == "takeover":
        classname = "to"
    elif folder.lower() == "not_takeover":
        classname = "not"
    else:
        classname = None

    m_ride = RIDE_RE.search(base)
    m_offset = OFFSET_RE.search(base)
    if not m_ride or not m_offset:
        print(f"Warning: could not parse m_ride or m_offset from filename {fname}, skipping.")
        return None
    ride = int(m_ride.group(1))
    offset = int(m_offset.group(1))
    m_take = TAKEOVER_EXPLICIT_RE.search(base)
    if m_take:
        takeover = int(m_take.group(1))
    else:
        m_num_before = NUM_BEFORE_OFFSET_RE.search(base)
        takeover = int(m_num_before.group(1)) if m_num_before else 0
    return ride, takeover, offset, classname

def make_mosaic_from_predictions(
    all_filenames,
    all_predictions,
    max_cols=25,
    gap=4,
    margin=20,
    border=2,
    text_height=16,
    bg=(255, 255, 255)
):
    """
    all_filenames: list of image file paths
    all_predictions: dict mapping filename to (classname, prob)
    Returns: PIL.Image mosaic
    """
    # Load images and parse metadata
    records = []
    for i, fname in enumerate(all_filenames):
        parsed = parse_from_name(fname)
        if not parsed:
            print(f"Warning: could not parse metadata from filename {fname}, skipping.")
            continue
        ride, takeover, offset, classname = parsed
        try:
            pil = Image.open(f"/scratch/tmp/p_scha35/cam-takeover-classification/data/split_classified/{fname}").convert("RGB")
        except Exception:
            print(f"Warning: could not open image {fname}, skipping.")
            continue
        records.append({
            "path": fname,
            "ride": ride,
            "takeover": takeover,
            "offset": offset,
            "pil": pil,
            "index": i,
            "classname": classname
        })

    if not records:
        raise ValueError("No valid images found.")

    # Ensure consistent tile size
    first_w, first_h = records[0]["pil"].size
    for r in records:
        if r["pil"].size != (first_w, first_h):
            r["pil"] = r["pil"].resize((first_w, first_h), Image.LANCZOS)

    # Group by (ride, takeover)
    groups = defaultdict(list)
    for r in records:
        print(f"Processing {r['path']} with index {r['index']}")
        pred = all_predictions[r["index"]]
        cname = pred == 1 and "to" or "not"
        groups[(r["ride"], r["takeover"])].append({
            "img": r["pil"],
            "offset": r["offset"],
            "classname": cname,
            "prob": 1.0,
            "ground_truth": r["classname"]
        })
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
        row_items = groups[key][:max_cols]
        for c, item in enumerate(row_items):
            x0 = margin + c * (cell_w + gap)
            y0 = margin + r * (cell_h + gap)
            outline_color = green if item["classname"] == item["ground_truth"] else red
            text = f"p:{item['classname']} t:{item['ground_truth']}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_x = x0 + (cell_w - text_w) // 2
            text_y = y0
            draw.text((text_x, text_y), text, fill=outline_color, font=font)
            img_y = y0 + text_height
            canvas.paste(item["img"], (x0 + border, img_y + border))
            for t in range(border):
                draw.rectangle(
                    [x0 + t, img_y + t, x0 + cell_w - 1 - t, img_y + cell_h - text_height - 1 - t],
                    outline=outline_color
                )
    return canvas