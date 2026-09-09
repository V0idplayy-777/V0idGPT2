"""Prepare CIFAR-10 (cloned GitHub mirror) for diffusion training.

Reads /home/user/imgdata/cifar10/{train,test}/<class>/*.png and writes:
  /home/user/imgdata/cifar_train.npz  (images uint8 [50000,32,32,3], labels)
  /home/user/imgdata/cifar_val.npz    (images uint8 [10000,32,32,3], labels)
"""
import os

import numpy as np
from PIL import Image

SRC = "/home/user/imgdata/cifar10"
DST = "/home/user/imgdata"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
# natural words used in training prompts (automobile -> car for most templates)
WORDS = {"airplane": "airplane", "automobile": "car", "bird": "bird",
         "cat": "cat", "deer": "deer", "dog": "dog", "frog": "frog",
         "horse": "horse", "ship": "ship", "truck": "truck"}
TEMPLATES = ["a photo of a {}", "a picture of a {}", "{}", "a {}",
             "a photo of the {}"]


def pack(split, name):
    imgs, labels = [], []
    base = os.path.join(SRC, split)
    for ci, c in enumerate(CLASSES):
        d = os.path.join(base, c)
        files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        for f in files:
            im = Image.open(os.path.join(d, f)).convert("RGB")
            assert im.size == (32, 32), f
            imgs.append(np.asarray(im, dtype=np.uint8))
            labels.append(ci)
    imgs = np.stack(imgs)
    labels = np.array(labels, dtype=np.uint8)
    np.savez(os.path.join(DST, name + ".npz"), images=imgs, labels=labels)
    print(f"{name}: {imgs.shape}, {len(np.unique(labels))} classes", flush=True)


def main():
    os.makedirs(DST, exist_ok=True)
    pack("train", "cifar_train")
    pack("test", "cifar_val")


if __name__ == "__main__":
    main()
