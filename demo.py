import cv2
import os
import matplotlib.pyplot as plt
from benchmark_tools.responsemap_mean_shift_triplet import benchmark_ms, loadmodel, resize
from demo.demo_utils import format_imshow, download_file_from_google_drive
import numpy as np


show_images = True

patch_size = 128
upscale_factor = 4
num_per_dim = 3200 / patch_size
upscale = True
embedding_size = 128

image_dir = "demo"
save_dir = "/heatmaps/"

model_ckpt_path = os.getcwd() + "/demo/hs_triplet_upscaling.model"
google_drive_id = "12ZBcH6L6qliOgvrHIzzakTqA2sVvsZiu"



if __name__=="__main__":
    if not os.path.exists(model_ckpt_path):
        print("please download model to {}".format(model_ckpt_path))
        print("Model does not exist, beginning model download to {}").format(os.path.join("demo", model_ckpt_path))
        download_file_from_google_drive(google_drive_id, model_ckpt_path)
    feat_extractor = loadmodel(model_ckpt_path)
    feat_extractor.eval()
    filenames = list(os.walk(image_dir + "/images"))[0][2]
    print("images read: {}".format(filenames))
    if not os.path.exists(image_dir + save_dir):
        os.makedirs(image_dir + save_dir)
        print("created folder {}".format(image_dir + save_dir))
    for j, suffix in enumerate(filenames):
        imname = image_dir + save_dir + suffix
        writename = os.path.splitext(imname)[0] + '.png'
        if not os.path.exists(writename):
            print("starting image " + str (suffix))
            imname = image_dir + "/images/" + suffix
            im = cv2.imread(imname)[:,:,::-1]
            if upscale:
                im = resize(im, upscale_factor*100)
            out_ms=benchmark_ms(im, feat_extractor)
            heatmap = (out_ms * 255).astype(np.uint8)
            heatmap = resize(heatmap, (1/float(upscale_factor))*100)
            cv2.imwrite(writename, heatmap)
            if show_images:
                f = plt.figure()
                f.add_subplot(1, 2, 1)
                plt.imshow(im)
                f.add_subplot(1, 2, 2)
                # th at
                plt.imshow(format_imshow(heatmap))
                plt.show()