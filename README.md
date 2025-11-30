# IAUN ImageProcessing EdgeDetection
## Image Processing Pipeline
Manual Implementation of Negative, Thresholding, Spatial Filters, and FFT

### 🚀Overview
This project implements a complete classical image processing pipeline without using built‑in filtering functions.

All operations are implemented manually with Python + NumPy, ensuring full control over pixel‑level computations.

The pipeline includes:

- Grayscale conversion
- Negative image transformation
- Thresholding (user-defined)
    + Manual spatial filtering:
    + Average filter (Low‑Pass)
- Laplacian filter (High‑Pass)
- 2D Fourier Transform (FFT), magnitude and phase visualization
- Zero Padding when required
- Displaying all outputs clearly

### 🧩 Detailed Features and Mathematical Implementation

**1. Grayscale Conversion (Luminance Method)**

        Conversion from an RGB image **I**<sub>rgb</sub> to a single-channel grayscale image **I**<sub>gray</sub> is performed using the standard ITU-R BT.601 luma definition, which accounts for human visual sensitivity to different color components.
        
                For a pixel **(_x, y_)** with color channels **R, G, B**:
                
                    **I**<sub>gray</sub>**(x, y)** = 0.299 . **_R_**(x, y) + 0.587 . **_G_**(x, y) + 0.114 . **_B_**(x, y)  
                    
                    This ensures the resulting grayscale image preserves perceived brightness correctly.

**2. Negative Transformation A pointwise intensity inversion:**

        The negative transformation inverts the intensity values of an image. If the maximum intensity level is  **L**<sub>max</sub> = 255:
        
        Negative(x, y) = L<sub>max</sub> - I(x, y) = 255 − I(x, y)

        This is crucial for visualizing subtle details often hidden in dark regions of an image.


**3. Thresholding Binary thresholding with user‑specified T:**

        Thresholding converts a grayscale image into a binary image based on a manually set threshold value (T).
        
        Thresholded(x,y)= 0 if I(x,y) < T    , 255 if I(x,y)≥T 
        


**4. Average Filter (Low‑Pass) Pure manual convolution with an N×N kernel (default: 3×3):**
        Spatial filtering is implemented via manual 2D convolution, where the image is scanned by a kernel (or mask).
        The output pixel value is the weighted sum of the neighborhood pixels determined by the kernel coefficients.

        **4.1. Average Filter (Low‑Pass Smoothing)**

                The Average filter is a smoothing filter used to reduce noise by averaging the intensities of neighboring pixels. For an (N x N) kernel:

                Kernel<sub>avg</sub>[i][j] = 1/N<sup>2</sup>  for all  i, j in [1,....,N ]
                
                For a standard 3x3 kernel, every element is (1/9).



        **4.2. Laplacian Filter (High‑Pass) Implemented manually using the classic kernel:**
            The Laplacian operator is a second-order derivative filter that highlights regions of rapid intensity change (edges). It is applied symmetrically around the center pixel. A common 3 x 3 implementation is:

            ``` 
             0  -1   0
            -1   4  -1
             0  -1   0
            ```

            The output of convolution with this kernel often requires renormalization or clipping to fit within the 0-255 range, as the results can be negative or exceed 255.
**5. Zero Padding (Border Handling and FFT Preparation)**
    Zero padding is essential for two primary reasons in this pipeline:

        **1- Boundary Conditions in Convolution:** To compute the filtered output pixel at the very edge of the image without losing information due to the kernel extending beyond the image boundary, the image matrix is typically padded with zeros.

        **2- FFT Resolution:** When preparing an image for the 2D FFT, zero-padding the input to dimensions that are powers of two (or significantly larger than the original image) can sometimes improve spectral resolution visualization or satisfy requirements for specific FFT implementations, although the core NumPy FFT handles non-power-of-two sizes efficiently. Padding to match the size of the convolution kernel is often necessary for accurate spatial domain filtering reconstruction via the frequency domain (Convolution Theorem).

**6. Fourier Transform (Frequency Domain Analysis)**

        The 2D Discrete Fourier Transform (DFT), computed efficiently using the 2D Fast Fourier Transform (FFT), transforms the image from the spatial domain **(x, y)** to the frequency domain **(u, v)**.

        The resulting complex output **F(u, v)** contains magnitude and phase information.

        **Magnitude Spectrum**

        The magnitude spectrum **M(u, v)** reveals the distribution of energy across different frequencies:
        
        M(u, v) = sqrt(Real(F(u, v))^2 + Image(F(u, v))^2)
        
         For visualization, the DC component (zero frequency) is typically shifted to the center using the FFT shift operation. 
         A logarithmic scaling is usually applied to compress the high dynamic range:
         
         Visualization = log(1 + M<sub>shifted</sub>(u, v)) 

        **Phase Spectrum**

        The phase spectrum **P(u, v)** determines the spatial localization of the features in the image:

        P(u, v) = arctan (Image(F(u, v)) / Real(F(u, v)))

---

**⚙️ Installation and Execution**

This project relies on standard scientific Python libraries for array manipulation and visualization.

**Prerequisites**

        Ensure you have Python 3.8 or newer installed.

        Required Libraries

        The following libraries must be installed into your environment:

        ```
        pip install numpy matplotlib pillow
        ```

        NumPy: Fundamental library for efficient array operations, critical for matrix manipulations in convolution and FFT.


        Matplotlib: Used for plotting and visualizing the input, intermediate, and final processed images.


        Pillow (PIL): Used for loading and saving standard image formats.

        Execution

        To run the entire pipeline with default settings (or settings defined within main.py):

        ```python main.py```


        The script is expected to load input_image.png, apply the sequential transformations, and display the results in separate windows or save them to disk.


**📌 Implementation Constraints and Notes**

This project adheres strictly to specific constraints imposed for educational rigor:


No Built‑in Filtering Functions: Functions like scipy.ndimage.convolve, scipy.signal.convolve2d, or cv2.filter2D are explicitly forbidden. The convolution operation must be built from scratch using basic NumPy array manipulation (slicing, element-wise multiplication, and summation).



Manual Operations: Every step, from calculating the luminosity weight in grayscale conversion to implementing the 2D FFT and subsequent magnitude/phase calculations, must be explicitly coded.



Academic Focus: The primary goal is demonstrating mastery of the underlying mathematical principles behind digital image processing algorithms, not achieving optimized production speed.

Convolution Implementation Detail

The manual convolution in filters.py must handle the creation of the padded image array first, then iterate through every pixel of the original image size, centering the kernel over the corresponding padded region, calculating the weighted sum, and placing the result in the output array.

### 📊Dataset
Dataset Reference
If you use the Berkeley Segmentation Dataset (BSDS300/BSDS500) in your experiments, please cite the original publication:
```
@InProceedings{MartinFTM01,
  author    = {D. Martin and C. Fowlkes and D. Tal and J. Malik},
  title     = {A Database of Human Segmented Natural Images and its Application
               to Evaluating Segmentation Algorithms and Measuring Ecological Statistics},
  booktitle = {Proc. 8th Int'l Conf. Computer Vision},
  year      = {2001},
  month     = {July},
  volume    = {2},
  pages     = {416--423}
}
```
Alternatively, in plain text:

Martin, D., Fowlkes, C., Tal, D., & Malik, J. (2001).

A Database of Human Segmented Natural Images and its Application to Evaluating Segmentation Algorithms and Measuring Ecological Statistics.

In Proc. ICCV, vol. 2, pp. 416–423.

### 📄 License
This project is intended for academic use. Redistribution or reuse must comply with university rules.