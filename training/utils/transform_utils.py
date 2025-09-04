import os
from PIL import Image
from torchvision.datasets import ImageFolder


class ImageFolderWithFilename(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        #print(f"path absolute: {path}")
        #print("root:",self.root)
        
        filename = os.path.basename(path)
        class_name = os.path.basename(os.path.dirname(path))
        split = os.path.basename(os.path.dirname(os.path.dirname(path)))
        sample.filename = "{}/{}/{}".format(split, class_name, filename)
        if self.transform:
            sample = self.transform(sample)
        return sample, target
class SaveEvery_nth_image:
    def __init__(self, n=10, save_dir="saved_images",name_saver=None):
        self.n = n
        self.name_saver = name_saver
        self.counter = 0
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, img):
        self.counter += 1
        read_name=self.name_saver.filename
        #print(f"Current image filename in SaveEvery_nth_image: {read_name}")
        if self.counter % self.n == 0:
            if (read_name is not None):
                image_name = read_name
            else:
                image_name = f"image_{self.counter}.png"
                
            img_path = os.path.join(self.save_dir, image_name)
            if not os.path.exists(os.path.dirname(img_path)):
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
            img.save(img_path)
        return img

class Image_Name_Saver:
    def __init__(self):
        self.filename = None
        
    def __call__(self, img):
        self.filename = img.filename
        #print(f"Image filename saved: {self.filename}")
        return img

class CenterCrop:
    
    def __init__(self,side="center", height=96, width=96):
        self.side = side
        self.height = height
        self.width = width
    def __call__(self, img):
        img_w, img_h = img.size
        th, tw = self.height, self.width
        
        if th > img_h:
            
            print(f"Warning : Crop size ({th,tw}) is larger than the image size {img_h,img_w}.")
            if hasattr(img, 'filename'):
                print(f"Image filename: {img.filename}")
        if tw > img_w:
            print(f"Warning : Crop size ({th,tw}) is larger than the image size {img_h,img_w}.")
            if hasattr(img, 'filename'):
                print(f"Image filename: {img.filename}")
        # Compute coordinates
        if self.side == "center":
            left = (img_w - tw) // 2
            top = (img_h - th) // 2
        elif self.side == "top":
            left = (img_w - tw) // 2
            top = 0
        elif self.side == "bottom":
            left = (img_w - tw) // 2
            top = img_h - th
        elif self.side == "left":
            left = 0
            top = (img_h - th) // 2
        elif self.side == "right":
            left = img_w - tw
            top = (img_h - th) // 2
        else:
            raise ValueError(f"Invalid side: {self.side}. Choose from 'center', 'top', 'bottom', 'left', 'right'.")

        right = left + tw
        bottom = top + th

        final_img = img.crop((left, top, right, bottom))
        return final_img
    
    