import os
from PIL import Image
import numpy as np


def patch(path, tosave, n):
    print(path)
    img = Image.open(path)
    img = np.array(img)

    name = path.split("/")[-1].split(".")[0]

    result = []
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
