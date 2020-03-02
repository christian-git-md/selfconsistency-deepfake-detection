from scipy.misc import imresize
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import rotate

crop = False
q = False


def process_data(im_a, im_b, label, augs=[], cv2=True):
    if q:
        rs = np.random.randint(0, 100)
        if not rs < 20:
            im_a = reresize(im_a, rs)
            im_b = reresize(im_b, rs)
    if cv2:
        im_a, im_b = (im_a[..., ::-1]), (im_b[..., ::-1])  # swap color channels
    if crop:
        im_a, im_b = random_crop(im_a, (64, 64)), random_crop(im_b, (64, 64))
    if augs:
        if label == 1:
            [im_a, im_b] = do_augments([im_a, im_b], augs)
        else:
            [im_a], [im_b] = do_augments([im_a], augs), do_augments([im_b], augs)
    im_a, im_b = im_a / 255., im_b / 255.
    im_a, im_b = to_pytorch(im_a), to_pytorch(im_b)
    return im_a, im_b, torch.tensor(label, dtype=torch.float32)


def do_augments(n_ims, augs):
    imsize = n_ims[0].shape[0]
    if 'resize' in augs:
        rs = np.random.randint(int(imsize * 0.5), int(imsize * 2))
        for i, im in enumerate(n_ims):
            n_ims[i] = imresize(im, (rs, rs))
            n_ims[i] = imresize(im, (imsize, imsize))
    if 'scale' in augs:
        scale_factor = np.random.randint(imsize + 1, int(imsize * 1.2))
        for i, im in enumerate(n_ims):
            n_ims[i] = scale_augmentation(im, scale_factor, crop_size=imsize)
    if 'blur' in augs:
        blur_factor = np.random.uniform(0, 0.5)
        for i, im in enumerate(n_ims):
            n_ims[i] = gaussian_filter(im, sigma=blur_factor)
    if 'rejpg' in augs:
        quality = np.random.uniform(80, 100)
        for i, im in enumerate(n_ims):
            n_ims[i] = rejpg(im, quality)
    if 'rotate' in augs:
        angle = np.random.randint(0, 180)
        for i, im in enumerate(n_ims):
            n_ims[i] = random_rotation(im, angle)
    for i, im in enumerate(n_ims):
        n_ims[i] = center_crop(im, (80, 80))
    return n_ims


def process_online_data(im, label, rs=None, cv2=True):
    if q and not rs < 20:
        for i, img in enumerate(im):
            im[i] = reresize(img, rs)
    if cv2:
        im = (im[..., ::-1])  # swap color channels
    if crop:
        new_im = np.zeros((8, 64, 64, 3))
        for i, j in enumerate(im):
            size = np.random.randint(65, 128)
            resized = resize(j, (size, size))
            new_im[i] = random_crop(resized, (64, 64))
        im = new_im
    im = im / 255.
    im = to_pytorch4d(im)
    return im, torch.tensor(label, dtype=torch.float32)


def scale_augmentation(image, scale_size, crop_size):
    image = imresize(image, (scale_size, scale_size))
    image = random_crop(image, crop_size)
    return image


def rejpg(image, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    image = cv2.imdecode(encimg, 1)
    return image


def reresize(image, quality):
    size = image.shape
    image = cv2.resize(image, (0, 0), fx=quality / 100., fy=quality / 100.)
    image = cv2.resize(image, (size[1], size[0]))
    return image


def random_crop(image, crop_size):
    crop_size = check_size(crop_size)
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image


def center_crop(image, crop_size):
    crop_size = check_size(crop_size)
    h, w, _ = image.shape
    top = h / 2 - crop_size[0] / 2
    left = w / 2 - crop_size[1] / 2
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image


def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size


def to_pytorch(im):
    return torch.tensor(np.transpose(im, (2, 0, 1)), dtype=torch.float32)


def to_pytorch4d(im):
    return torch.tensor(np.transpose(im, (0, 3, 1, 2)), dtype=torch.float32)


def noisy(image, var):
    image = image.astype(np.uint16)
    print(image)
    row, col, ch = image.shape
    mean = 0
    sigma = var ** 0.5
    gauss = np.floor(255 * np.random.normal(mean, sigma, (row, col, ch)))
    gauss = gauss.reshape(row, col, ch)
    # print(image, gauss)
    noisy = np.min([np.ones(image.shape) * 255, image + gauss], axis=0)
    noisy = noisy.astype(np.uint8)
    # print(noisy)
    # print("noisy", gauss)
    print(noisy)
    return noisy


def random_rotation(image, angle, reshape=False):
    h, w, _ = image.shape
    image = rotate(image, angle, reshape=reshape)
    image = center_crop(image, (h, w))
    return image


def resize(image, size):
    size = check_size(size)
    image = imresize(image, size)
    return image
