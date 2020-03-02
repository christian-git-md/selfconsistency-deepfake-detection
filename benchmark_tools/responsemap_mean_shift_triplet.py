import sys
sys.path.insert(0, '..')
import cv2
import numpy as np
from network import OnlineFeatureNetwork
import torch
from tqdm import tqdm
import scipy

patch_size = 128
embedding_size = 128
num_per_dim = 3200 / patch_size


def get_coordinate_pairs(h, w):
    coordinate_pairs = []
    for y in range(0, h):
        for x in range(0, w):
            coordinate_pairs.append((y, x))
    return coordinate_pairs


def getpatch(coordinate_pair, image, real_coordinates_h, real_coordinates_w):
    h = real_coordinates_h[coordinate_pair[0]]
    w = real_coordinates_w[coordinate_pair[1]]
    patch = image[h:h + patch_size, w:w + patch_size, :]
    return patch


def resize(image, quality):
    image = cv2.resize(image, (0, 0), fx=quality / 100., fy=quality / 100.)
    return image


def loadmodel(model_weight_path):
    model = OnlineFeatureNetwork(resnet_type="rn18", num_outputs=embedding_size)
    model.cuda()
    model.load_state_dict(torch.load(model_weight_path))
    return model


def format_pytorch(image):
    torch_image = (torch.tensor(np.transpose(image / 255., (2, 0, 1)), dtype=torch.float32)).unsqueeze(0)
    return torch_image


def mean_shift(points_, heat_map, iters=5):
    points = np.copy(points_)
    kdt = scipy.spatial.cKDTree(points)
    eps_5 = np.percentile(scipy.spatial.distance.cdist(points, points, metric='euclidean'), 10)

    for epis in range(iters):
        for point_ind in range(points.shape[0]):
            point = points[point_ind]
            nearest_inds = kdt.query_ball_point(point, r=eps_5)
            points[point_ind] = np.mean(points[nearest_inds], axis=0)
    val = []
    for i in range(points.shape[0]):
        val.append(kdt.count_neighbors(scipy.spatial.cKDTree(np.array([points[i]])), r=eps_5))
    ind = np.nonzero(val == np.max(val))
    return np.mean(points[ind[0]], axis=0).reshape(heat_map.shape[0], heat_map.shape[1])


def preextract_features(patch, feat_extractor):
    with torch.no_grad():
        preextract_features = feat_extractor(patch.unsqueeze(0))
    return preextract_features


def benchmark_ms(im, feat_extractor):
    stride = (max(im.shape[0], im.shape[1]) - patch_size) // num_per_dim
    max_h_ind = int(np.floor((im.shape[0] - patch_size) / float(stride)))
    max_w_ind = int(np.floor((im.shape[1] - patch_size) / float(stride)))
    spread = max(1, patch_size // stride)
    real_coordinates_h = (np.linspace(start=0, stop=im.shape[0] - patch_size, num=max_h_ind)).astype(np.int)
    real_coordinates_w = (np.linspace(start=0, stop=im.shape[1] - patch_size, num=max_w_ind)).astype(np.int)
    all_coordinates = get_coordinate_pairs(max_h_ind, max_w_ind)
    all_anchors = get_coordinate_pairs(max_h_ind, max_w_ind)
    frames = np.zeros((max_h_ind, max_w_ind, max_h_ind, max_w_ind))
    count_nonzero = np.zeros((max_h_ind, max_w_ind, max_h_ind, max_w_ind))
    preextracted_features = torch.tensor(np.zeros((max_h_ind, max_w_ind, embedding_size))).float()
    for coord in all_coordinates:
        patch = getpatch(coord, im, real_coordinates_h, real_coordinates_w)
        preextracted_features[coord[0], coord[1], :] = preextract_features(format_pytorch(patch).cuda(), feat_extractor)
    print(preextracted_features.shape)
    for coords_a in tqdm(all_anchors):
        for i, coords_b in enumerate(all_coordinates):
            input = (
            (preextracted_features[coords_a[0], coords_a[1], :], preextracted_features[coords_b[0], coords_b[1], :]))
            dist = (torch.norm(input[0] - input[1], p=2)).data.cpu().numpy()
            prob = 1 - dist / 2.
            frames[coords_a[0]: (coords_a[0] + spread),
            coords_a[1]: (coords_a[1] + spread),
            coords_b[0]: (coords_b[0] + spread),
            coords_b[1]: (coords_b[1] + spread)] += prob
            count_nonzero[coords_a[0]: (coords_a[0] + spread),
            coords_a[1]: (coords_a[1] + spread),
            coords_b[0]: (coords_b[0] + spread),
            coords_b[1]: (coords_b[1] + spread)] += 1
    res = frames / (count_nonzero + 0.00001)
    ms = mean_shift(res.reshape((-1, res.shape[0] * res.shape[1])), res)
    if np.mean(ms > .5) > .5:
        ms = 1 - ms
    out_ms = cv2.resize(ms, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
    return out_ms
