from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from tqdm import tqdm
from utils import process_online_data
import pickle
from utils import rejpg


class Live_Online_Dataset(Dataset):
    """
    Gives a fixed number of patches from a single picture, which later have to be assembled to triplets in-memory
    """
    def __init__(self, rootdir, preprocess=[], patchsize = 128, num_per_im =8, check_corrupt=False):
        check_corrupt = False
        self.rootdir = rootdir
        self.preprocess = preprocess
        self.filenames = list(os.walk(rootdir))[0][2]
        if check_corrupt:
            self.filenames = self.check_corrupt(self.filenames)
        self.len = len(self.filenames)
        self.patchsize = patchsize
        self.num_per_im = num_per_im
        print(self.len)
        return

    def __getitem__(self, index):
        patches = self.get_n_patches(index)
        return process_online_data(patches, np.expand_dims(np.array(index), -1), cv2=True)

    def __len__(self):
        return self.len


    def get_n_patches(self, index):
        image = self.read_im(index)
        np.random.seed(seed = int(index // 64))
        rs = np.random.randint(0,100)
        if rs > 30:
            image = rejpg(image, rs)
        patches = np.zeros((self.num_per_im, self.patchsize, self.patchsize, 3), dtype=np.uint8)
        for i in range(0, self.num_per_im):
            patches[i,:,:,:] = self.rand_patch(image)
        return patches

    def rand_patch(self, image):
        h = np.random.randint(image.shape[0] - self.patchsize + 1)
        w = np.random.randint(image.shape[1] - self.patchsize + 1)
        return image[h:h + self.patchsize, w:w + self.patchsize, :]

    def read_im(self, index):
        while True:
            try:
                return self.try_imread(index)
            except (UnicodeEncodeError, TypeError, ValueError):
                index = np.random.randint(self.len)

    def try_imread(self, index):
        im = cv2.imread(self.rootdir + "/" + self.filenames[index])
        if type(im) == np.ndarray and im.shape[0]>128 and im.shape[1]>128:
            return im
        else:
            raise TypeError


    def check_corrupt(self, raw_filenames):
        filenames = []
        try:
            return pickle.load(open(self.rootdir+"/clean_images.pickle", "rb"))
        except IOError:
            print("checking for corrupt files, this might take a while")
            for file in tqdm(raw_filenames):
                try:
                    if type(cv2.imread(self.rootdir + "/" + file)) == np.ndarray:
                        filenames.append(file)
                except UnicodeEncodeError:
                    continue
            pickle.dump(filenames, open(self.rootdir+"/clean_images.pickle", "wb"))
            return filenames