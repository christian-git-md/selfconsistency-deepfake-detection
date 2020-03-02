import torchvision
import torch
from torch import nn
import os
from datasets.hdf5_dataset import HDF5_Dataset
from datasets.hdf5_online_dataset import HDF5_Online_Dataset
import time


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant(m.bias, 0)


def freeze_all(model):
    for child in model.children():
            for param in child.parameters():
                param.requires_grad = False
    return model


def get_dataset_from_type(dataset_type, rootdir, num_train_samples, preprocess):
    if dataset_type == 'hdf5':
        dataset = HDF5_Dataset(rootdir, num_train_samples, preprocess=preprocess)
    elif dataset_type == 'hdf5-online':
        dataset = HDF5_Online_Dataset(rootdir, num_train_samples, preprocess=preprocess)
    else:
        raise TypeError("invalid dataset type")
    return dataset


def get_resnet(resnet_type, pretrained):
    if resnet_type == 'rn18':
        resnet = torchvision.models.resnet18(pretrained=pretrained)
    elif resnet_type == 'rn34':
        resnet = torchvision.models.resnet34(pretrained=pretrained)
    elif resnet_type == 'rn50':
        resnet = torchvision.models.resnet50(pretrained=pretrained)
    elif resnet_type == 'rn101':
        resnet = torchvision.models.resnet101(pretrained=pretrained)
    elif resnet_type == 'rn152':
        resnet = torchvision.models.resnet152(pretrained=pretrained)
    else:
        raise ValueError('invalid resnet type')
    return resnet

def save_models(step, best_model):
    best_model=best_model.state_dict()
    if not os.path.exists(os.getcwd() + "/ckpt"):
        os.makedirs(os.getcwd() + "/ckpt")
    torch.save(best_model, "ckpt/step_{}.model".format(step))
    print("Checkpoint saved")

def get_dict_value_or_none(dict, key):
    try:
        return dict[key]
    except KeyError:
        return None

class Triplet_Selector():
    def __init__(self, num_per_im, margin = 0.2):
        self.num_per_im = num_per_im
        self.margin = margin


    def select_triples(self, all_outputs):
        with torch.no_grad():
            distances = self.get_distances(all_outputs)
            triplets = self.get_semi_hard_negative_pairs(all_outputs, distances)
            return triplets


    def get_distances(self, all_outputs):
        """
        Repeats outputs along new dimension and calculates all the pairwise distances
        """
        num_outpus = all_outputs.shape[0]
        # repeat outputs along new dim
        repeated_output = all_outputs.repeat(num_outpus, 1).view(num_outpus, num_outpus, -1)
        # repeat each element of outputs
        repeated_elementwise = all_outputs.reshape(num_outpus, 1, -1).repeat(1 ,num_outpus, 1).view(num_outpus, num_outpus, -1)
        squared_dist = torch.norm(repeated_output-repeated_elementwise, p=2, dim=2) ** 2
        return squared_dist


    def get_semi_hard_negative_pairs(self, all_outputs, distances):
        '''
        Finds the semi-hard samples based on l2 distances, faithfully to facenet paper https://arxiv.org/abs/1503.03832
        but instead parallelized for GPU efficiency

        :return: The relevant batch indices of the semi-hard samples
        '''
        t = time.time()
        num_outpus = all_outputs.shape[0]
        num_images = num_outpus / self.num_per_im
        image_distances = distances.view(
            num_outpus/self.num_per_im, self.num_per_im, -1) # 4 x 16 x 64
        pos_indices = torch.eye(num_images, device='cuda', dtype=torch.uint8).view(
            num_images, 1, num_images, 1).repeat(
            1, self.num_per_im, 1, self.num_per_im).view(
            num_outpus, num_outpus
        )
        positive_dist = torch.masked_select(distances, pos_indices)
        positive_distances = positive_dist.view(num_images, self.num_per_im, self.num_per_im) # 4 x 16 x 16
        num_distances_per_anchor = image_distances.shape[2]
        num_pos_distances = positive_distances.shape[2]
        # repeat positive distances and all distances to the same shape,
        # leaving the anchor patch dimension invariant in both cases
        repeated_pos = positive_distances.unsqueeze(3).repeat(
            1, 1, 1, num_distances_per_anchor) # 4 x 16 x 16 - > 4 x 16 x 16 x 64
        repeated_dist = image_distances.unsqueeze(2).repeat(
            1, 1, num_pos_distances, 1) # 4 x 16 x 64 -> 4 x 16 x 16 x 64
        same_image = pos_indices.view(
            num_images , self.num_per_im , 1, num_distances_per_anchor).repeat(
            1, 1, self.num_per_im ,1)
        semi_hard = (repeated_pos < repeated_dist) & (repeated_pos + self.margin > repeated_dist) & (same_image ^ 1)
        hard_indices = semi_hard.view(num_images * self.num_per_im, self.num_per_im, num_images * self.num_per_im).nonzero()
        hard_indices[:, 1] += (hard_indices[:, 0] // self.num_per_im) * self.num_per_im
        return hard_indices