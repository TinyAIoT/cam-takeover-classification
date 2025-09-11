import os
from PIL import Image
from torchvision.datasets import ImageFolder

class ImageFolderWithFilenameReturn(ImageFolder):
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
        return sample, target, "{}/{}/{}".format(split, class_name, filename)