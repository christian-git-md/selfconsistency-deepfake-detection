from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from tqdm import tqdm
from utils import process_data
import pickle


class Live_Dataset(Dataset):
    """
    Creates pairs of patches from the same or different image and the respective lables from an imagefolder
    """
    def __init__(self, rootdir, preprocess=[], patchsize = 128, check_corrupt=True):
        self.rootdir = rootdir
        self.filenames = []
        self.preprocess = preprocess
        files = list(os.walk(rootdir))[0][2]
        if check_corrupt:
            self.filenames=self.check_corrupt(files)
            # print(self.filenames)
        else:
            self.filenames = files
        self.len = len(self.filenames)
        self.patchsize = patchsize
        return

    def __getitem__(self, index):
        im_a, im_b, label = self.random_pair(index)
        label = np.expand_dims(np.array(label), -1)
        return process_data(im_a, im_b, label, self.preprocess)

    def __len__(self):
        return self.len


    def random_pair(self, index):
        im_a = self.read_im(index)
        patch_a = self.rand_patch(im_a)
        if np.random.randint(2) > 0.5:
            patch_b = self.rand_patch(im_a)
            label = 1
        else:
            im_b = self.read_im(np.random.randint(self.len))
            patch_b = self.rand_patch(im_b)
            label = 0
        return patch_a, patch_b, label


    def read_im(self, index):
        while True:
            try:
                return self.try_imread(index)
            except (UnicodeEncodeError, TypeError, ValueError):
                index = np.random.randint(self.len)


    def try_imread(self, index):
        im = cv2.imread(self.rootdir + "/" + self.filenames[index])
        # print(im, self.rootdir + "/" + self.filenames[index])
        if type(im) == np.ndarray and im.shape[0]>128 and im.shape[1]>128:
            return im
        else:
            raise TypeError


    def rand_patch(self, image):
        h = np.random.randint(image.shape[0] - self.patchsize + 1)
        w = np.random.randint(image.shape[1] - self.patchsize + 1)
        return image[h:h + self.patchsize, w:w + self.patchsize, :]

    def check_corrupt(self, raw_filenames):
        filenames = []
        try:
            return pickle.load(open(self.rootdir + "/clean_images.pickle", "rb"))
        except IOError:
            print("checking for corrupt files, this might take a while")
            for file in tqdm(raw_filenames):
                try:
                    if type(cv2.imread(self.rootdir + "/" + file)) == np.ndarray:
                        filenames.append(file)
                except UnicodeEncodeError:
                    continue
            pickle.dump(filenames, open(self.rootdir + "/clean_images.pickle", "wb"))
            return filenames


# dataset = Live_Dataset("/home/chrissikek/MA/Forgery_classifier/flickrdownloader/tag_2019")
# for i in range(0,200):
#     a = dataset[i][0]
#     #print(a.shape)