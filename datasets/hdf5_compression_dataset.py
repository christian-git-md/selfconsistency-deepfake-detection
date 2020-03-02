from torch.utils.data import Dataset
import h5py
import torch
import numpy as np
from matplotlib import pyplot as plt
from utils import process_online_data


class HDF5_Compression_Dataset(Dataset):
    """
    Reads an hdf5 file which contains multiple patches of one image at any given index,
    which later can be assembled to triplets in-memory
    """
    def __init__(self, rootdir, len, preprocess=[], num_per_im = 8):
        self.num_per_im = num_per_im
        self.rootdir = rootdir
        self.len = len
        self.preprocess = preprocess
        self.patchsize = 128
        return

    def __getitem__(self, index):
        file = h5py.File(self.rootdir + "train" + str(index // 100000) + ".hdf5", "r")
        patches = self.getimhdf5(file, index % 100000)
        # resize augmentation
        np.random.seed(seed = int(index // 64))
        rs = np.random.randint(0,100)
        return patches

    def __len__(self):
        return self.len

    def getimhdf5(self, hdf5_file, a):
        patches = np.zeros((self.num_per_im, self.patchsize, self.patchsize, 3), dtype=np.uint8)
        for i in range(0, self.num_per_im):
            patches[i,:,:,:] = hdf5_file["train_im_"+str(i)][a]
        return patches


        # dataset = HDF5_Dataset("/media/chrissikek/hdd/sctf/preextractedhdf5/train/", 1000000)
        # print(dataset[44656][0].shape)
