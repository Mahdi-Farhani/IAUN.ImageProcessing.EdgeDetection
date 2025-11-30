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

def convolution2d(image:np.array, kernel:np.array):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    output = np.zeros_like(image)

    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)

    return output

def average_filter(image:np.array, filter_size:int=3):
    kernel = np.ones((filter_size, filter_size), dtype=np.float32) / (filter_size * filter_size)
    blurred= convolution2d(image, kernel)
    return np.clip(blurred, 0, 255).astype(np.uint8)

def laplacian_filter(image:np.array):
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]], dtype=np.float32)
    laplacian= convolution2d(image, kernel)
    
    laplacian=laplacian-laplacian.min()
    laplacian=laplacian*(255/laplacian.max())

    return laplacian.astype(np.uint8)

def fourier_transform(image:np.array):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    magnitude = np.abs(fshift)
    phase = np.angle(fshift)

    magnitude_log = np.log(1 + magnitude)

    return fshift, magnitude_log, phase


if __name__ == "__main__":
    filename="25098"
    (originalImage,rgb)=loadImage(f"../samples/{filename}.jpg")
    gray = rgb_to_gray(rgb)
    negative = negative_image(gray)
    threshold= int(input("Enter threshold value (0-255): "))

    binary = thresholding_binary(gray, threshold=threshold)

    binary_otsu, otsu_threshold=apply_otsu_thresholding(gray)

    print("Otsu Threshold =", otsu_threshold)


    average = average_filter(gray, filter_size=3)
    laplacian = laplacian_filter(gray)

    F_shift, magnitude_log, phase = fourier_transform(gray)


    plt.figure(figsize=(20,12))
    plt.subplot(3, 4, 1); plt.title("Original"); plt.imshow(rgb.astype(np.uint8)); plt.axis('off')
    plt.subplot(3, 4, 2); plt.title("Grayscale"); plt.imshow(gray, cmap='gray'); plt.axis('off')
    plt.subplot(3, 4, 3); plt.title("Negative"); plt.imshow(negative, cmap='gray'); plt.axis('off')
    plt.subplot(3, 4, 4); plt.title(f"Binary (t:{threshold})"); plt.imshow(binary, cmap='gray'); plt.axis('off')    

    plt.subplot(3, 4, 5); plt.title("Histogram"); plt.hist(gray.flatten(), bins=256)
    plt.subplot(3, 4, 6); plt.title("Average Filter (Low-Pass)"); plt.imshow(average, cmap='gray'); plt.axis('off')
    plt.subplot(3, 4, 7); plt.title("Laplacian Filter (High-Pass)"); plt.imshow(laplacian, cmap='gray'); plt.axis('off')
    plt.subplot(3, 4, 8); plt.title(f"Otsu Binary (t:{otsu_threshold})"); plt.imshow(binary_otsu, cmap='gray'); plt.axis('off')    

    plt.subplot(3, 4, 9); plt.title("FFT Magnitude (log)"); plt.imshow(magnitude_log, cmap='gray'); plt.axis('off')
    plt.subplot(3, 4, 10); plt.title("FFT Phase"); plt.imshow(phase, cmap='gray'); plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"../outputs/image_processing_results_{filename}.png")
    plt.show()