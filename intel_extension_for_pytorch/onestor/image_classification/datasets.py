# Write a custom dataset class (inherits from torch.utils.data.Dataset)
import torch
#import torchvision.io as io
import io
from torch.utils.data import Dataset
from PIL import Image
import pathlib
from typing import Dict, List, Tuple

from torchvision.datasets.folder import find_classes

import accimage

# resolve this by using _init_.py instead
#################################################
import sys
sys.path.insert(0, '../third_party/oneStorage/src/python/build/')
from _pywrap_oneFile import oneFile
#print(dir(oneFile))
#################################################

def onestor_loader(path: str) -> torch.Tensor:
    #return io.read_image(path, io.image.ImageReadMode.RGB).type(torch.float32)
    #print(path)
    data = oneFile().read(path)
    #print(data)
    img_tensor = torch.frombuffer(data, dtype=torch.uint8)
    #return io.decode_image(img_tensor, io.image.ImageReadMode.RGB).type(torch.float32)

# 1. Subclass torch.utils.data.Dataset
class ImageFolderOneStor(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.JPEG")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)
        #self.reader = oneFile()
        self.reader = oneFile(64*1024, 16, True, False, 1)

    # 4. Make function to load images
    def load_image(self, index: int) -> torch.Tensor:
        "Reads an image via a path and decode it."
        #print(self.paths[index])
        #data = self.reader.read(str(self.paths[index]))
        filename = str(self.paths[index])
        num_bytes = self.reader.get_file_size(filename)
        
        #img_tensor = torch.rand(num_bytes).type(torch.uint8)
        #return self.reader.sync_tensor_read(img_tensor, filename)#.type(torch.float32)

        #return accimage.Image(filename)

        #-----------method with file read and then decode -----------#
        data = bytearray(num_bytes)
        self.reader.read(filename,data)
        return Image.open(io.BytesIO(data)).convert("RGB")
        #---------------------end-------------------------------------#
        
        #data = bytearray(num_bytes)
        #self.reader.sync_pread(filename, data)
        #img_tensor = torch.frombuffer(data, dtype=torch.uint8)
        #return io.decode_image(img_tensor, io.image.ImageReadMode.RGB).type(torch.float32)
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)