from torch.utils.data import Dataset
import h5py
import numpy as np
from utils import process_data


class HDF5_Dataset(Dataset):
    """
    Reads image pairs from hdf5 files
    """
    def __init__(self, rootdir, len, preprocess=[]):
        self.rootdir = rootdir
        self.len = len
        self.preprocess = preprocess
        self.filelen = 100000
        return

    def __getitem__(self, index):
        file = h5py.File(self.rootdir + "train" + str(index // self.filelen) + ".hdf5", "r")
        im_a, im_b, label = self.getimhdf5(file, index % self.filelen)
        if type(label) == np.uint8:
            label = np.expand_dims(label, -1)
        return process_data(im_a, im_b, label, self.preprocess)

    def __len__(self):
        return self.len

    def getimhdf5(self, hdf5_file, a):
        im = hdf5_file["train_im_a"][a]
        im2 = hdf5_file["train_im_b"][a]
        label = hdf5_file["train_labels"][a]
        return im, im2, label
