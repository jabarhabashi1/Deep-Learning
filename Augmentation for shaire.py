"""
******* Spectral Augmentation code *******
MIT License

Copyright (c) 2025 Jabar_Habashi
This code belongs to the manuscript titled _Revealing critical mineralogical insights in extreme environments
using deep learning technique on hyperspectral PRISMA satellite imagery: Dry Valleys, South Victoria Land, Antarctica._

authors: Jabar Habashi,, Amin Beiranvand Pour, Aidy M Muslim, Ali Moradi Afrapoli, Jong Kuk Hong, Yongcheol Park,
Alireza Almasi, Laura Crispini, Mazlan Hashim and Milad Bagheri
Journal: ISPRS Journal of Photogrammetry and Remote Sensing
DOI:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Code"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
2. Any use of the Software must include a citation of the manuscript titled _Deep Learning Integration of ASTER
and PRISMA Satellite Imagery for Alteration Mineral Mapping in Dry Valley, South Victoria Land, Antarctica._

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.

"""


import numpy as np

input_file = "rC:\\drive\\SL.txt"         # Import ASCL spectral library file path. note: use the envi classic ASCL file
try:
    # Reading data from a file
    data = np.loadtxt(input_file, skiprows=1)  # Remove header (if any)
    wavelengths = data[:, 0]  # First column: wavelengths
    reflectances = data[:, 1:]  # Next columns: Reflections for minerals

    # Parameters for Offset
    initial_offset = 0  # Starting offset
    increment = 0.01  # Amount to add in each iteration
    num_iterations = 30  # Total number of iterations

    # Prepare to store all results
    all_augmented_data = []

    # Apply Offset in iterations
    for i in range(num_iterations):
        current_offset = initial_offset + i * increment
        augmented_reflectances = reflectances + current_offset
        all_augmented_data.append(augmented_reflectances)

    # Stack all results horizontally with wavelengths
    all_augmented_data = np.hstack([wavelengths[:, None]] + all_augmented_data)

    # Save new data to output file
    output_file = "Vegetation.txt"
    with open(output_file, "w") as f:

        for row in all_augmented_data:
            f.write(" ".join(f"{value:.6f}" for value in row) + "\n")

    print(f"The Spectral Profiles is saved as '{output_file}'.")

except FileNotFoundError:
    print(f"File '{input_file}' not found. Please make sure you entered the file path correctly.")
except Exception as e:
    print(f"An error occurred: {e}")