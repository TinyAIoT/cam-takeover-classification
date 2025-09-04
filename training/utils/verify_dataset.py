from PIL import Image
import os
import argparse

def ends_with_img_ext(filename):
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    return filename.lower().endswith(img_extensions)
def verify_folder(folder):
    bad_files = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if (ends_with_img_ext(f)):
                path = os.path.join(root, f)
                try:
                    with Image.open(path) as img:
                        img.verify()  # prüft, ob Header konsistent ist
                except Exception as e:
                    print(f"Corrupted file: {path} ({e})")
                    bad_files.append(path)

    # Alle kaputten Dateien löschen
    for bf in bad_files:
        os.remove(bf)
        print(f"Gelöscht: {bf}")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifies and cleans image dataset folders.")
    parser.add_argument("--folder", type=str, help="Path to image folder")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"The given path is not a folder: {args.folder}")
    else:
        verify_folder(args.folder)