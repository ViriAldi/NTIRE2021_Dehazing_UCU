import os

import numpy as np
import yaml
from PIL import Image


def read_config(path):
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.join(*[str(i) for i in seq])

    yaml.add_constructor('!join', join)

    with open(path) as f:
        return yaml.load(f.read().replace('\t', ' '))


def patch(path, tosave, n):
    print(path)
    img = Image.open(path)
    img = np.array(img)

    name = path.split("/")[-1].split(".")[0]

    x = img.shape[0] // n
    y = img.shape[1] // n

    for i in range(n):
        for j in range(n):
            new = img[i * x : i * x + x, j * y : j * y + y, :]
            Image.fromarray(new).save(os.path.join(tosave, f"{name}_{i+1}_{j+1}.png"))


def convert(path, output, n):
    if not os.path.exists(output):
        os.makedirs(output)

    for name in os.listdir(path):
        patch(os.path.join(path, name), output, n)


def rescale(path1, path2, new_size):
    img = Image.open(path1)
    img = img.resize(new_size, Image.LANCZOS)
    img.save(path2)


def rescale_dataset(path1, path2, new_size):
    gt_path0 = os.path.join(path1, "GT")
    hazy_path0 = os.path.join(path1, "hazy")

    gt_path = os.path.join(path2, "GT")
    hazy_path = os.path.join(path2, "hazy")
    if not os.path.exists(gt_path):
        os.makedirs(gt_path)
    if not os.path.exists(hazy_path):
        os.makedirs(hazy_path)

    for filename in os.listdir(gt_path0):
        rescale(os.path.join(gt_path0, filename), os.path.join(gt_path, filename), new_size)
    
    for filename in os.listdir(hazy_path0):
        rescale(os.path.join(hazy_path0, filename), os.path.join(hazy_path, filename), new_size)
