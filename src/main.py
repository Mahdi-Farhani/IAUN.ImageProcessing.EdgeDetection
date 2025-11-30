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

    gray = 0.299 * R + 0.587 * G + 0.114 * B
    return gray.astype(np.uint8)

def negative_image(gray:np.array):
    negative = 255 - gray
    return negative.astype(np.uint8)

def thresholding_binary(gray:np.array, threshold:int=128):
    height, width = gray.shape
    binary = np.zeros((height, width), dtype=np.uint8)

    binary = np.where(gray >= threshold, 255, 0)
    return binary.astype(np.uint8)

if __name__ == "__main__":
    (originalImage,rgb)=loadImage("../samples/22013.jpg")
    gray = rgb_to_gray(rgb)
    negative = negative_image(gray)
    threshold= int(input("Enter threshold value (0-255): "))

    binary = thresholding_binary(gray, threshold=threshold)

    plt.figure(figsize=(15,4))
    plt.subplot(1, 4, 1); plt.title("Original"); plt.imshow(rgb.astype(np.uint8)); plt.axis('off')
    plt.subplot(1, 4, 2); plt.title("Grayscale"); plt.imshow(gray, cmap='gray'); plt.axis('off')
    plt.subplot(1, 4, 3); plt.title("Negative"); plt.imshow(negative, cmap='gray'); plt.axis('off')
    plt.subplot(1, 4, 4); plt.title(f"Binary (t:{threshold})"); plt.imshow(negative, cmap='gray'); plt.axis('off')

    plt.show()