#!/usr/bin/env python3
"""
Image labelling GUI:
- Scans two top-level folders "mostly_takeover" and "mostly_not_takeover" (each has rides -> manoeuvers -> images)
- For each manoeuver, takes the FIRST 16 images and builds a 96x96 mosaic:
    * crop left 240x240 of each 320x240 image
    * resize to 24x24
    * paste into a 4x4 grid (row-major: 1st top-left, 2nd = row1 col2, 5th = row2 col1, etc.)
- Displays the mosaic; press:
    0 -> save to data/classified/takeover
    1 -> save to data/classified/not_takeover
- Quits when all mosaics are labeled
"""

import os
import re
import csv
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, filedialog

# ---------- Configuration ----------
# You can set ROOT_DIR explicitly (folder that contains "mostly_takeover" and "mostly_not_takeover"),
# or leave as None to pick via dialog at start (defaults to script's dir if canceled).
ROOT_DIR = "/home/paula/Documents/reedu/TinyAIoT/cam-takeover-detection/data/real_raw"  # e.g., Path("/path/to/data/root")

TOP_FOLDERS = ["mostly_takeover", "mostly_not_takeover"]  # the two top-level folders to walk
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

OUTPUT_ROOT = Path("data/classified")
CLASS_KEY_TO_SUBFOLDER = {
    "0": "takeover",  # pressing 0 -> class "takeover"
    "1": "not_takeover",  # pressing 1 -> class "not_takeover"
}

TILE_CROP_SIZE = (240, 240)   # width, height after cropping (left square)
TILE_OUT_SIZE  = (24, 24)     # width, height after scaling a single tile
GRID_DIM       = (4, 4)       # 4x4 grid
CANVAS_SIZE    = (GRID_DIM[0]*TILE_OUT_SIZE[0], GRID_DIM[1]*TILE_OUT_SIZE[1])  # 96x96

# ---------- Helpers ----------
_num_re = re.compile(r"(\d+)")

def natural_key(s: str):
    """Sort key that tries to split numbers for natural ordering."""
    parts = _num_re.split(s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def list_dirs(path: Path):
    return sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name))

def list_images(path: Path):
    imgs = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return sorted(imgs, key=lambda p: natural_key(p.name))

def build_mosaic_from_first_16(img_paths):
    """
    Builds the 96x96 mosaic image from the first 16 images.
    Crop: take right 240x240 from a 320x240 image -> box (80,0,320,240)
    Then resize to 24x24 and paste on a 4x4 grid.
    """
    if len(img_paths) < 16:
        return None  # not enough images, skip this manoeuver

    # prepare blank canvas
    mosaic = Image.new("RGB", CANVAS_SIZE, (0, 0, 0))

    for idx in range(16):
        p = img_paths[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            # crop right square 240x240 (from x=80 to x=320)
            crop_box = (80, 0, 320, 240)
            tile = im.crop(crop_box)
            # resize to 24x24
            tile = tile.resize(TILE_OUT_SIZE, Image.BILINEAR)

        # compute grid position
        row = idx // GRID_DIM[0]
        col = idx %  GRID_DIM[0]
        x = col * TILE_OUT_SIZE[0]
        y = row * TILE_OUT_SIZE[1]
        mosaic.paste(tile, (x, y))

    return mosaic

def ensure_output_dirs():
    for cls in CLASS_KEY_TO_SUBFOLDER.values():
        (OUTPUT_ROOT / cls).mkdir(parents=True, exist_ok=True)

def unique_save_path(base_dir: Path, base_name: str, ext: str = ".png") -> Path:
    candidate = base_dir / f"{base_name}{ext}"
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = base_dir / f"{base_name}_{i:03d}{ext}"
        if not candidate.exists():
            return candidate
        i += 1

def init_csv_log():
    """Initialize the CSV file with headers if it doesn't exist."""
    csv_path = OUTPUT_ROOT / "labels.csv"
    if not csv_path.exists():
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Create header with path_img_1 through path_img_16, then label
            header = [f"path_img_{i+1}" for i in range(16)] + ["label"]
            writer.writerow(header)
    return csv_path

def log_to_csv(csv_path: Path, img_paths: list, label: str, root_dir: Path):
    """Append a row to the CSV with the 16 image paths and label."""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # Convert Path objects to relative strings
        paths_str = [str(p.relative_to(root_dir)) for p in img_paths]
        row = paths_str + [label]
        writer.writerow(row)

def get_already_labeled_groups(csv_path: Path, root_dir: Path):
    """
    Read the CSV and return a set of tuples representing already labeled groups.
    Each tuple: (ride, top, manoeuver, offset)
    """
    if not csv_path.exists():
        return set()
    
    labeled_groups = set()
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        # Skip header
        next(reader, None)
        
        for row in reader:
            if len(row) >= 17:  # 16 paths + 1 label
                # Extract info from first image path
                first_img_path = Path(root_dir) / row[0]
                if first_img_path.exists():
                    parts = Path(row[0]).parts
                    if len(parts) >= 3:
                        ride = parts[0]
                        top = parts[1]
                        manoeuver = parts[2]
                        
                        # Get the maneuver directory and list all images
                        man_dir = root_dir / ride / top / manoeuver
                        if man_dir.exists():
                            all_imgs = list_images(man_dir)
                            
                            # Find the offset by locating the first image in the sorted list
                            first_img_name = first_img_path.name
                            try:
                                offset = next(i for i, img in enumerate(all_imgs) if img.name == first_img_name)
                                labeled_groups.add((ride, top, manoeuver, offset))
                            except StopIteration:
                                # Image not found in current list, skip this entry
                                continue
    
    return labeled_groups

def filter_unlabeled_groups(groups, labeled_groups):
    """
    Filter out groups that have already been labeled.
    """
    unlabeled = []
    for group in groups:
        group_key = (group["ride"], group["top"], group["manoeuver"], group["offset"])
        if group_key not in labeled_groups:
            unlabeled.append(group)
    
    return unlabeled

# ---------- Data Gathering ----------
def collect_manoeuver_groups(root_dir: Path):
    """
    Returns a list of dicts:
        {
          'ride': ride_name,
          'top': 'mostly_takeover' or 'mostly_not_takeover',
          'manoeuver': manoeuver_name,
          'images': [Path, ...]   # 16 images starting from this offset
          'offset': int           # starting index for this group
        }
    Creates overlapping groups of 16 images from each maneuver.
    """
    items = []
    for ride_dir in list_dirs(root_dir):
        for top in TOP_FOLDERS:
            top_path = ride_dir / top
            if not top_path.is_dir():
                continue
            for man_dir in list_dirs(top_path):
                all_imgs = list_images(man_dir)
                
                # Create overlapping groups of 16 images
                if len(all_imgs) >= 16:
                    for start_idx in range(len(all_imgs) - 15):  # -15 ensures we always have 16 images
                        group_imgs = all_imgs[start_idx:start_idx + 16]
                        items.append({
                            "ride": ride_dir.name,
                            "top": top,
                            "manoeuver": man_dir.name,
                            "images": group_imgs,
                            "offset": start_idx
                        })
    return items

def auto_label_not_takeover_groups(groups, csv_path, root_dir):
    """
    Automatically label all groups from 'mostly_not_takeover' folders.
    Returns the count of auto-labeled groups.
    """
    auto_labeled_count = 0
    ensure_output_dirs()
    
    for group in groups:
        if group["top"] == "mostly_not_takeover":
            # Create mosaic and save
            mosaic = build_mosaic_from_first_16(group["images"])
            if mosaic is not None:
                # Save mosaic
                out_dir = OUTPUT_ROOT / "not_takeover"
                offset = group.get("offset", 0)
                base = f"{group['ride']}_{group['top']}_{group['manoeuver']}_offset{offset:03d}"
                base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
                out_path = unique_save_path(out_dir, base, ".png")
                mosaic.save(out_path)
                
                # Log to CSV
                log_to_csv(csv_path, group["images"], "not_takeover", root_dir)
                auto_labeled_count += 1
    
    return auto_labeled_count

# ---------- GUI App ----------
class LabellerApp:
    def __init__(self, root_dir: Path, unlabeled_groups, all_groups, already_labeled_count):
        self.root_dir = root_dir
        self.all_groups = all_groups  # Keep original list for progress tracking
        self.groups = unlabeled_groups  # Only the groups that need labeling
        
        ensure_output_dirs()
        self.csv_path = init_csv_log()
        
        self.index = 0
        self.current_mosaic = None
        self.current_meta = None
        
        # Use the pre-calculated count
        self.already_labeled = already_labeled_count

        self.root = tk.Tk()
        self.root.title("Manoeuver Labeller (0→class takeover, 1→class not_takeover)")

        # Info label
        self.info_var = tk.StringVar()
        self.info_label = tk.Label(self.root, textvariable=self.info_var, font=("TkDefaultFont", 10))
        self.info_label.pack(padx=10, pady=(10, 4), anchor="w")

        # Auto-suggest label (for mostly_takeover)
        self.auto_var = tk.StringVar()
        self.auto_label = tk.Label(self.root, textvariable=self.auto_var, font=("TkDefaultFont", 10, "italic"), fg="#0f0")
        self.auto_label.pack(padx=10, pady=(0, 4), anchor="w")

        # Progress label
        self.progress_var = tk.StringVar()
        self.progress_label = tk.Label(self.root, textvariable=self.progress_var, font=("TkDefaultFont", 9), fg="#666")
        self.progress_label.pack(padx=10, pady=(0, 4), anchor="w")

        # Canvas to show the 96x96 image, upscale for visibility (optional)
        self.display_scale = 4  # show 384x384 but save at 96x96
        self.canvas = tk.Canvas(self.root,
                                width=CANVAS_SIZE[0]*self.display_scale,
                                height=CANVAS_SIZE[1]*self.display_scale,
                                bg="#222")
        self.canvas.pack(padx=10, pady=10)

        # Hint label
        hint = "Press 0 → takeover, 1 → not_takeover.  q to quit."
        self.hint_label = tk.Label(self.root, text=hint, fg="#555")
        self.hint_label.pack(padx=10, pady=(0, 10))

        # Key bindings
        self.root.bind("0", self.key_class)
        self.root.bind("1", self.key_class)
        self.root.bind("q", self.quit_app)
        self.root.bind("<Escape>", self.quit_app)

        if self.groups:
            self.show_next()
        else:
            self.show_all_done()

    def render_info(self):
        if self.current_meta is None:
            self.info_var.set("Done.")
            self.auto_var.set("")
            self.progress_var.set("")
            return
        meta = self.current_meta
        nimgs = len(meta["images"])
        offset = meta.get("offset", 0)
        text = (f"[{self.index+1}/{len(self.groups)}]  Ride: {meta['ride']}  "
                f"Top: {meta['top']}  Manoeuver: {meta['manoeuver']}  "
                f"Images: {nimgs} (offset: {offset})")
        self.info_var.set(text)
        
        # Show auto-suggested label for mostly_takeover
        if meta["top"] == "mostly_takeover":
            self.auto_var.set("Auto-suggested label: takeover (press 'a' to accept)")
        else:
            self.auto_var.set("")
        
        # Show progress including already labeled
        total_done = self.already_labeled + self.index
        total_groups = len(self.all_groups)
        progress_text = f"Progress: {total_done}/{total_groups} ({self.already_labeled} previously labeled, {len(self.groups) - self.index} remaining)"
        self.progress_var.set(progress_text)

    def show_all_done(self):
        self.current_meta = None
        self.current_mosaic = None
        self.render_info()
        self.canvas.delete("all")
        
        if self.already_labeled > 0:
            msg = f"All done! Total groups labeled: {len(self.all_groups)} ({self.already_labeled} were already labeled from previous sessions)"
        else:
            msg = "All done! No more image groups to label."
        
        messagebox.showinfo("All done", msg)
        self.root.after(200, self.root.destroy)

    def show_next(self):
        # All groups now have exactly 16 images, so no need to check
        if self.index < len(self.groups):
            meta = self.groups[self.index]
            imgs = meta["images"]
            self.current_meta = meta
            self.render_info()

            mosaic = build_mosaic_from_first_16(imgs)
            self.current_mosaic = mosaic
            self.display_on_canvas(mosaic)
        else:
            # We're done
            self.show_all_done()

    def display_on_canvas(self, pil_img: Image.Image):
        # Upscale for display only
        disp = pil_img.resize((pil_img.width * self.display_scale, pil_img.height * self.display_scale), Image.NEAREST)
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw")

    def save_current(self, class_key: str):
        if self.current_mosaic is None or self.current_meta is None:
            return
        
        # Determine the label to use
        if class_key == "a":
            # Use auto-assigned label for mostly_takeover
            label = "takeover"
        else:
            # Use manual override
            label = CLASS_KEY_TO_SUBFOLDER[class_key]
        
        # Find the subfolder for this label
        sub = label
        out_dir = OUTPUT_ROOT / sub

        meta = self.current_meta
        offset = meta.get("offset", 0)
        # Construct a descriptive base name including offset
        base = f"{meta['ride']}_{meta['top']}_{meta['manoeuver']}_offset{offset:03d}"
        # sanitize filename a bit
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)

        out_path = unique_save_path(out_dir, base, ".png")
        self.current_mosaic.save(out_path)
        
        # Log to CSV with relative paths
        log_to_csv(self.csv_path, meta["images"], label, self.root_dir)
        
        # Move to next
        self.index += 1
        self.show_next()

    def auto_label_current(self, event=None):
        """Shortcut method to auto-label current item."""
        self.save_current("a")

    def key_class(self, event):
        if event.char in CLASS_KEY_TO_SUBFOLDER:
            self.save_current(event.char)

    def quit_app(self, event=None):
        self.root.destroy()

    def run(self):
        self.root.mainloop()

# ---------- Main ----------
def main():
    global ROOT_DIR
    if ROOT_DIR is None:
        # Let user choose a directory that contains ride folders with mostly_takeover and mostly_not_takeover
        root = filedialog.askdirectory(title="Select folder that contains ride directories (Cancel = current dir)")
        if root:
            ROOT_DIR = Path(root)
        else:
            ROOT_DIR = Path.cwd()

    all_groups = collect_manoeuver_groups(Path(ROOT_DIR))
    if not all_groups:
        messagebox.showerror("No data", f"No manoeuvers found under {ROOT_DIR}\nExpected structure: ride_*/mostly_takeover|mostly_not_takeover/*/images")
        return

    # Check if there are any unlabeled groups
    ensure_output_dirs()
    csv_path = OUTPUT_ROOT / "labels.csv"
    init_csv_log()
    
    labeled_groups = set()
    if csv_path.exists():
        labeled_groups = get_already_labeled_groups(csv_path, Path(ROOT_DIR))
    
    # Filter out already labeled groups
    unlabeled_groups = filter_unlabeled_groups(all_groups, labeled_groups)
    
    # Auto-label all "mostly_not_takeover" groups
    not_takeover_groups = [g for g in unlabeled_groups if g["top"] == "mostly_not_takeover"]
    if not_takeover_groups:
        messagebox.showinfo("Auto-labeling", f"Auto-labeling {len(not_takeover_groups)} groups from 'mostly_not_takeover' folders...")
        auto_labeled_count = auto_label_not_takeover_groups(not_takeover_groups, csv_path, Path(ROOT_DIR))
        messagebox.showinfo("Auto-labeling Complete", f"Auto-labeled {auto_labeled_count} groups as 'not_takeover'")
    else:
        auto_labeled_count = 0
    
    # Only show groups from "mostly_takeover" for manual labeling
    manual_groups = [g for g in unlabeled_groups if g["top"] == "mostly_takeover"]
    
    if not manual_groups:
        total_auto_labeled = len([g for g in all_groups if g["top"] == "mostly_not_takeover"]) - len([g for g in not_takeover_groups])
        total_labeled = len(labeled_groups) + auto_labeled_count
        messagebox.showinfo("All Complete", f"All groups processed!\nTotal: {len(all_groups)}\nPreviously labeled: {len(labeled_groups)}\nAuto-labeled this session: {auto_labeled_count}\nCheck {csv_path} for results.")
        return
    
    # Show manual labeling interface for mostly_takeover groups only
    total_already_labeled = len(labeled_groups) + auto_labeled_count
    app = LabellerApp(Path(ROOT_DIR), manual_groups, all_groups, total_already_labeled)
    app.run()

if __name__ == "__main__":
    main()
