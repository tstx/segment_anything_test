from webcam import webcam_capture
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import glob
import time

def show_anns(anns,save_path):
    if len(anns) == 0:
        print(save_path)
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
    plt.savefig(save_path, dpi=100)
        

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam_checkpoint = "./sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
inpath = "./data/in"
outpath = "./data/out"

webcam_capture(inpath)

print("loading model...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
print("model loaded")

mask_generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")

files = []
files.extend(glob.glob(fr"{inpath}/*.png"))
files.extend(glob.glob(fr"{inpath}/*.jpg"))
idx = 0
for file in files:
    img = cv2.imread(file)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"image.shape:{img.shape}")
    plt.clf()
    plt.subplot(1,2,1)
    plt.imshow(img)

    plt.subplot(1,2,2)
    plt.imshow(img)
    print(f"generating masks idx: {idx}")
    start = time.time()
    masks = mask_generator.generate(img)
    print(time.time()-start)
    print(fr"masks:{len(masks)}")
    print(masks[0].keys())
    show_anns(masks,fr"{outpath}/{idx}.png")
    idx += 1

    # plt.show()


# for i, mask_data in enumerate(masks):
#     mask = mask_data["segmentation"]
#     img = np.ones((mask.shape[0], mask.shape[1], 3))
#     color_mask = np.random.random((1, 3)).tolist()[0]
#     for i in range(3):
#         img[:,:,i] = color_mask[i]
#     cv2.imshow('Output', np.dstack((img, mask*0.35)))
#     cv2.waitKey(0)

