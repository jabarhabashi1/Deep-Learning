"""
******* AVCA endmember extractor code *******

MIT License

Copyright (c) 2025 Jabar_Habashi
This code belongs to the manuscript titled _Revealing critical mineralogical insights in extreme environments
using deep learning technique on hyperspectral PRISMA satellite imagery: Dry Valleys, South Victoria Land, Antarctica._

authors: Jabar Habashi,, Amin Beiranvand Pour, Aidy M Muslim, Ali Moradi Afrapoli, Jong Kuk Hong, Yongcheol Park,
Alireza Almasi, Laura Crispini, Mazlan Hashim and Milad Bagheri
Journal: ISPRS Journal of Photogrammetry and Remote Sensing
DOI:https://doi.org/10.1016/j.isprsjprs.2025.07.005

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Code"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
2. Any use of the Software must include a citation of the article: _Habashi, Jabar, Amin B. Pour, Aidy M. Muslim, Ali M. Afrapoli, Jong K. Hong, Yongcheol Park,
Alireza Almasi, Laura Crispini, Mazlan Hashim, and Milad Bagheri. "Revealing Critical Mineralogical Insights in Extreme Environments Using Deep Learning Technique
on Hyperspectral PRISMA Satellite Imagery: Dry Valleys, South Victoria Land, Antarctica."
ISPRS Journal of Photogrammetry and Remote Sensing 228, (2025): 83-121. https://doi.org/10.1016/j.isprsjprs.2025.07.005._

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.

also, The primary code is a Python translation of the original MATLAB implementation
by Jose Nascimento and Jose Bioucas Dias.
While the original MATLAB code does not explicitly specify copyright, we have further improved and adapted
the implementation in this project.
The primary Python version can be found at: https://github.com/Laadr/VCA/blob/master/VCA.py.


For more details on the algorithm, refer to:
J. M. P. Nascimento and J. M. B. Dias, "Vertex component analysis: a fast algorithm to unmix hyperspectral data,"
IEEE Transactions on Geoscience and Remote Sensing, vol. 43, no. 4, pp. 898-910, April 2005,
doi: 10.1109/TGRS.2005.844293.

***Read Line 200-203, and 218 For Importing The Data***
"""

import sys
import numpy as np
import rasterio
import os
import re

# Internal functions
def estimate_snr(Y, r_m, x):
    [L, N] = Y.shape  # L number of bands (channels), N number of pixels
    [p, N] = x.shape  # p number of endmembers (reduced dimension)

    P_y = np.sum(Y ** 2) / float(N)
    P_x = np.sum(x ** 2) / float(N) + np.sum(r_m ** 2)
    snr_est = 10 * np.log10((P_x - p / L * P_y) / (P_y - P_x))

    return snr_est

def vca(Y, R, verbose=True, snr_input=0):

    # Initializations
    if len(Y.shape) != 2:
        sys.exit('Input data must be of size L (number of bands i.e. channels) by N (number of pixels)')

    [L, N] = Y.shape  # L number of bands (channels), N number of pixels

    R = int(R)
    if R < 0 or R > L:
        sys.exit('ENDMEMBER parameter must be integer between 1 and L')


    # SNR Estimates
    if snr_input == 0:
        y_m = np.mean(Y, axis=1, keepdims=True)
        Y_o = Y - y_m  # data with zero-mean
        Ud = np.linalg.svd(np.dot(Y_o, Y_o.T) / float(N))[0][:, :R]  # computes the R-projection matrix
        x_p = np.dot(Ud.T, Y_o)  # project the zero-mean data onto p-subspace

        SNR = estimate_snr(Y, y_m, x_p)

        if verbose:
            print("SNR estimated = {}[dB]".format(SNR))
    else:
        SNR = snr_input
        if verbose:
            print("input SNR = {}[dB]\n".format(SNR))

    SNR_th = 15 + 10 * np.log10(R)


    # Choosing Projective Projection or projection to p-1 subspace
    if SNR < SNR_th:
        if verbose:
            print("... Select proj. to R-1")

        d = R - 1
        if snr_input == 0:  # it means that the projection is already computed
            Ud = Ud[:, :d]
        else:
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m  # data with zero-mean

            Ud = np.linalg.svd(np.dot(Y_o, Y_o.T) / float(N))[0][:, :d]  # computes the p-projection matrix
            x_p = np.dot(Ud.T, Y_o)  # project thezeros mean data onto p-subspace

        Yp = np.dot(Ud, x_p[:d, :]) + y_m  # again in dimension L

        x = x_p[:d, :]  # x_p =  Ud.T * Y_o is on a R-dim subspace
        c = np.amax(np.sum(x ** 2, axis=0)) ** 0.5
        y = np.vstack((x, c * np.ones((1, N))))
    else:
        if verbose:
            print("... Select the projective proj.")

        d = R
        Ud = np.linalg.svd(np.dot(Y, Y.T) / float(N))[0][:, :d]  # computes the p-projection matrix

        x_p = np.dot(Ud.T, Y)
        Yp = np.dot(Ud, x_p[:d, :])  # again in dimension L (note that x_p has no null mean)

        x = np.dot(Ud.T, Y)
        u = np.mean(x, axis=1, keepdims=True)  # equivalent to  u = Ud.T * r_m
        y = x / np.dot(u.T, x)

    # VCA algorithm

    indice = np.zeros((R), dtype=int)
    A = np.zeros((R, R))
    A[-1, 0] = 1

    for i in range(R):
        w = np.random.rand(R, 1)
        f = w - np.dot(A, np.dot(np.linalg.pinv(A), w))
        f = f / np.linalg.norm(f)

        v = np.dot(f.T, y)

        indice[i] = np.argmax(np.absolute(v))
        A[:, i] = y[:, indice[i]]  # same as x(:,indice(i))

    Ae = Yp[:, indice]

    return Ae, indice, Yp

# Reading Raster Data

def read_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read()  # Reads data in (bands, rows, cols) format
        wavelengths = src.tags().get('Wavelengths', None)  # Example of accessing metadata
        if wavelengths:
            wavelengths = list(map(float, wavelengths.split(',')))
        else:
            wavelengths = list(range(1, data.shape[0] + 1))  # Default to band indices
    L, rows, cols = data.shape
    Y = data.reshape(L, rows * cols)  # Convert to 2D array (bands x pixels)
    return Y, rows, cols, wavelengths

def read_mask(mask_path, rows, cols):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # Assuming the mask is single-band
        if mask.shape != (rows, cols):
            raise ValueError("Mask dimensions do not match the raster dimensions.")
    return mask.reshape(-1)  # Flatten the mask


# Extract Wavelengths from HDR File

def extract_wavelengths(hdr_path):
    wavelengths = []
    with open(hdr_path, 'r') as file:
        content = file.read()
        match = re.search(r'wavelength\s*=\s*{([^}]*)}', content)
        if match:
            wavelengths = [float(wave.strip()) for wave in match.group(1).split(',')]
    return wavelengths


# Saving Results
def save_results(output_dir, wavelengths, Ae):
    output_file = os.path.join(output_dir, "AVCA_results.txt")
    with open(output_file, "w") as f:
        # Save wavelengths in the first column
        f.write("Wavelength, " + ", ".join([f"Endmember_{i+1}" for i in range(Ae.shape[1])]) + "\n")
        for i, wavelength in enumerate(wavelengths):
            f.write(f"{wavelength}, " + ", ".join(map(str, Ae[i, :])) + "\n")

    print(f"Results saved to {output_file}")
    print("Extracted Endmembers:")
    print("Wavelengths", "\t" + "\t".join([f"Endmember_{i+1}" for i in range(Ae.shape[1])]))
    for i, wavelength in enumerate(wavelengths):
        print(f"{wavelength}\t" + "\t".join(map(str, Ae[i, :])))


"""**************************Importing the RS Data**************************"""

if __name__ == "__main__":
    raster_file = "import RS data path"  # Path to your raster file
    mask_file = "import mask data path"  # Path to your mask file
    hdr_file = "import RS data path"  # Path to your HDR file
    output_dir = os.path.dirname(raster_file)

    # Read the raster data and mask
    Y, rows, cols, wavelengths = read_raster(raster_file)
    mask = read_mask(mask_file, rows, cols)

    # Extract wavelengths from HDR file
    extracted_wavelengths = extract_wavelengths(hdr_file)

    # Apply the mask
    Y = Y[:, mask == 1]  # Only process pixels where the mask is 1
    if Y.shape[1] == 0:
        raise ValueError("No valid pixels remain after applying the mask.")

    R = "import the number" # Number of endmembers to extract
    Ae, indice, Yp = vca(Y, R)

    save_results(output_dir, extracted_wavelengths, Ae)
