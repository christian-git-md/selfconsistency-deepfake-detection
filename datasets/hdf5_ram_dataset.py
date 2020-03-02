from torch.utils.data import Dataset
import numpy as np
import tables
from utils import process_data


class HDF5_Ram_Dataset(Dataset):
    """
    This version of the hdf5 Dataloader loads the whole file into ram to limit number of hard drive operations
    """
    def __init__(self, rootdir, len, preprocess=[]):
        self.rootdir = rootdir
        self.len = len
        self.current_index = -1
        self.current_file = tables.open_file(self.rootdir + "train0.hdf5", driver="H5FD_CORE")
        self.preprocess = preprocess
        return

    def __getitem__(self, index):
        file_index = str(index // 100000)
        if file_index != self.current_index:
            self.current_file.close()
            self.current_index = file_index
            self.current_file = tables.open_file(self.rootdir + "train" + file_index + ".hdf5", driver="H5FD_CORE")
        im_a, im_b, label = self.getimhdf5(index % 100000)
        if type(label) == np.uint8:
            label = np.expand_dims(label, -1)
        return process_data(im_a, im_b, label, self.preprocess)

    def __len__(self):
        return self.len

    def getimhdf5(self, a):
        im = self.current_file.root.train_im_a[a]
        im2 = self.current_file.root.train_im_b[a]
        label = self.current_file.root.train_labels[a]
        return im, im2, label
