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
def otsu_thresholding(gray:np.array):
    histogram, _ = np.histogram(gray, bins=256, range=(0, 256))
    total_pixels = gray.size

    current_max = 0
    threshold = 0
    sum_total = np.dot(np.arange(256), histogram)
    sum_background = 0
    weight_background = 0

    for t in range(256):
        weight_background += histogram[t]
        if weight_background == 0:
            continue

        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * histogram[t]

        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = t

    return threshold

def apply_otsu_thresholding(gray:np.array):
    threshold = otsu_thresholding(gray)
    binary = thresholding_binary(gray, threshold)
    return binary, threshold

if __name__ == "__main__":
    (originalImage,rgb)=loadImage("../samples/22013.jpg")
    gray = rgb_to_gray(rgb)
    negative = negative_image(gray)
    threshold= int(input("Enter threshold value (0-255): "))

    binary = thresholding_binary(gray, threshold=threshold)

    binary_otsu, otsu_threshold=apply_otsu_thresholding(gray)

    print("Otsu Threshold =", otsu_threshold)


    plt.figure(figsize=(18,8))
    plt.subplot(2, 4, 1); plt.title("Original"); plt.imshow(rgb.astype(np.uint8)); plt.axis('off')
    plt.subplot(2, 4, 2); plt.title("Grayscale"); plt.imshow(gray, cmap='gray'); plt.axis('off')
    plt.subplot(2, 4, 3); plt.title("Negative"); plt.imshow(negative, cmap='gray'); plt.axis('off')
    plt.subplot(2, 4, 4); plt.title(f"Binary (t:{threshold})"); plt.imshow(binary, cmap='gray'); plt.axis('off')    
    plt.subplot(2, 4, 5); plt.title("Histogram"); plt.hist(gray.flatten(), bins=256)
    plt.subplot(2, 4, 8); plt.title(f"Otsu Binary (t:{otsu_threshold})"); plt.imshow(binary_otsu, cmap='gray'); plt.axis('off')    


    plt.show()