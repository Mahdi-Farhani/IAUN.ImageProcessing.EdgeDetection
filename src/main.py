import numpy as np
from PIL import Image as image

import matplotlib.pyplot as plt

def loadImage(path:str):
    img = image.open(path)
    rgb = np.array(img, dtype=np.float32)
    return (img,rgb)

def rgb_to_gray(rgb:np.array):
    R = rgb[:,:,0]
    G = rgb[:,:,1]
    B = rgb[:,:,2]

    gray = 0.299 * R + 0.687 * G + 0.114 * B
    return gray.astype(np.uint8)


if __name__ == "__main__":
    (originalImage,rgb)=loadImage("../samples/22013.jpg")
    gray = rgb_to_gray(rgb)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.title("Original"); plt.imshow(rgb.astype(np.uint8)); plt.axis('off')
    plt.subplot(1,2,2); plt.title("Grayscale"); plt.imshow(gray, cmap='gray'); plt.axis('off')
    plt.show()