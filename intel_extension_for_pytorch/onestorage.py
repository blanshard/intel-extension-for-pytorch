# coding: utf-8
import torch
from torch.utils.data import IterableDataset, Dataset
import torch.distributed as dist
from itertools import chain

# resolve this by using _init_.py instead
#################################################
import sys
sys.path.insert(0, '../csrc/oneStorage/build')
from _pywrap_oneFile import oneFile
#################################################

class oneFileDataset(Dataset):
    """A mapped-style dataset.
    """
    def __init__(self, root):
        """
        Args:
            root (root of the dataset)
        """
        # Initialize the handler
        self.handler = oneFile()
        self.filenames = self.handler.list_files(root)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filedata = self.handler.read(filename)
        return filename, filedata

class oneFileIterableDataset(IterableDataset):
    """Iteraterable dataset.
    """
    def __init__(self, root, shuffle=False):
        self.epoch   = 0
        self.shuffle = shuffle
        self.dist = dist.is_initialized() if dist.is_available() else False
        if self.dist:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()

        # Initialize the handler
        self.handler = oneFile()
        self.filenames = self.handler.list_files(root)

    @property
    def shuffled_list(self):
        if self.shuffle:
            random.seed(self.epoch)
            return random.sample(self.filenames, len(self.filenames))
        else:
            return self.filenames

    def get_data(self, filename):
        yield filename, self.handler.read(filename)

    def get_stream(self, filenames):
        return chain.from_iterable(map(self.get_data, filenames))

    def worker_dist(self, filenames):
        if self.dist:
            total_size = len(filenames)
            filenames = filenames[self.rank:total_size:self.world_size]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            wid = worker_info.id
            num_workers = worker_info.num_workers
            length = len(filenames)
            return filenames[wid:length:num_workers]
        else:
            return filenames

    def __iter__(self):
        filenames = self.worker_dist(self.shuffled_list)
        return self.get_stream(filenames)

    def __len__(self):
        return len(self.filenames)

    def set_epoch(self, epoch):
        self.epoch = epoch
