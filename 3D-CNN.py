"""
******* CNN code *******
MIT License

Copyright (c) 2025 Jabar_Habashi
This code belongs to the manuscript titled _Revealing critical mineralogical insights in extreme environments
using deep learning technique on hyperspectral PRISMA satellite imagery: Dry Valleys, South Victoria Land, Antarctica._

authors: Jabar Habashi,, Amin Beiranvand Pour, Aidy M Muslim, Ali Moradi Afrapoli, Jong Kuk Hong, Yongcheol Park,
Alireza Almasi, Laura Crispini, Mazlan Hashim and Milad Bagheri
Journal: ISPRS Journal of Photogrammetry and Remote Sensing
https://doi.org/10.1016/j.isprsjprs.2025.07.005.

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

Note for Leveraging the Code
To effectively utilize this code, a robust spectral library is essential.
Strong spectral data ensures accurate results and optimal performance.
Additionally, implementing spectral augmentation is highly recommended to enhance the analysis
and improve the generalizability of the model.

***See Augmentation code.***

***Read Line 86, 87 ,108, 147-157 For Importing The Data***
"""

import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from spectral import open_image, imshow
from spectral import envi
from spectral.io.envi import read_envi_header
from spectral.io.envi import open as open_envi
import warnings
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
import os

warnings.filterwarnings("ignore")

# Custom Callback for metrics logging
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        pass  # No metrics calculation during intermediate epochs

# Updated read_envi_header for debugging
def debug_read_envi_header(file):
    print(f"Reading header file: {file}")
    try:
        with open(file, 'r', encoding='utf-8') as f:
            if not f.readline().strip().startswith('ENVI'):
                raise Exception('Missing "ENVI" keyword in the header.')
            lines = f.readlines()
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError: {e}. File might be binary or have wrong encoding.")
        raise
    except Exception as e:
        print(f"Unexpected error while reading header: {e}")
        raise
    return read_envi_header(file)

def load_RS_data():
    try:
        print("Loading Remote Sensing data...")
        RS_data = open_image("import RS data path").load()                 #Import your RS data EX: r"C:\\drive\\rs.hdr"
        with open("import HDR data path") as HDR:                      #Import your RS HDR data EX: r"C:\\drive\\rs.hdr"
            print("Remote Sensing data loaded successfully.")
            print(f"Remotely Sensed data shape: {RS_data.shape}")
            lines = HDR.readlines()
            map_info = None
            coordinate_system_string = None
            for line in lines:
                if 'map info' in line.lower():
                    map_info = line.strip().split('=', 1)[-1].strip()
                elif 'coordinate system string' in line.lower():
                    coordinate_system_string = line.strip().split('=', 1)[-1].strip()
            print("Map Info:", map_info)
            print("Coordinate System String:", coordinate_system_string)
            return RS_data, map_info, coordinate_system_string
    except Exception as e:
        print(f"Error loading Remote Sensing data: {e}")
        raise

def load_mask():
    try:
        print("Loading mask...")
        mask = open_image("import mask path").load()                        #Import your mask EX: r"C:\\drive\\mask.hdr"
        print("Mask loaded successfully.")
        print(f"Mask shape: {mask.shape}")
        return mask
    except Exception as e:
        print(f"Error loading mask: {e}")
        raise
def apply_mask_to_data(RS_data, mask):
    try:
        print("Applying mask to data...")
        RS_data_flat = RS_data.reshape(-1, RS_data.shape[2])
        mask_flat = mask.flatten()
        valid_pixels = mask_flat == 1
        RS_data_masked = RS_data_flat[valid_pixels]
        print(f"Masked data shape: {RS_data_masked.shape}")
        return RS_data_masked, valid_pixels
    except Exception as e:
        print(f"Error applying mask to data: {e}")
        raise

# Load spectral libraries for each mineral
def load_spectral_library_with_header(sli_path, hdr_path):
    try:
        print(f"Loading spectral library and header: {sli_path}, {hdr_path}")
        spectral_lib = open_envi(hdr_path)
        print("Spectral library loaded successfully.")
        print(f"Spectral shape: {spectral_lib.spectra.shape}")
        return spectral_lib
    except Exception as e:
        print(f"Error loading spectral library: {e}")
        raise

# Load spectral libraries for multiple minerals
def load_all_spectral_libraries():
    try:
        print("Loading all spectral profiles library...")

        # Dictionary for storing spectral profiles data and labels
        # import Spectral profiles data and labels EX:
        #spectral_libraries = {
            #"mineral 1": load_spectral_library_with_header(
                #r"drive path\\mineral 1",
                #r"drive path\\mineral 1.hdr"),
            #...
            #...
            #...
            #"mineral N": load_spectral_library_with_header(
                #r"drive path\\mineral N",
                #r"drive path\\mineral N.hdr")
        }
        print("All spectral profiles library loaded successfully.")

        # Extract spectral data and labels
        spectral_data = np.vstack([lib.spectra for lib in spectral_libraries.values()])
        labels = np.concatenate([
            np.full(len(lib.spectra), idx) for idx, lib in enumerate(spectral_libraries.values())
        ])

        # Extract class names
        class_labels = list(spectral_libraries.keys())

        return spectral_data, labels, class_labels

    except Exception as e:
        print(f"Error loading spectral libraries: {e}")
        return None, None, None

# Reshape Remote Sensing and Spectral Profiles data for CNN
def reshape_data_for_cnn(X, name="data"):
    try:
        print(f"Reshaping {name} for 3D CNN...")
        num_samples = X.shape[0]  # Number of samples
        band_count = X.shape[1]  # Number of bands

        # Reconstruct the data into a form suitable for 3D CNN input
        reshaped = X.reshape((num_samples, 1, 1, band_count, 1))  # Use all bands on the input channel
        print(f"{name} reshaped to {reshaped.shape}.")
        return reshaped
    except Exception as e:
        print(f"Error reshaping {name}: {e}")
        raise

class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, accuracy_threshold, loss_threshold):
        super(CustomEarlyStopping, self).__init__()
        self.accuracy_threshold = accuracy_threshold
        self.loss_threshold = loss_threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_acc = logs.get('accuracy', 0)
        val_acc = logs.get('val_accuracy', 0)
        train_loss = logs.get('loss', float('inf'))
        val_loss = logs.get('val_loss', float('inf'))

        if (train_acc >= self.accuracy_threshold and
                val_acc >= self.accuracy_threshold and
                train_loss <= self.loss_threshold and
                val_loss <= self.loss_threshold):
            print(f"Stopping training at epoch {epoch + 1} as thresholds are met.")
            self.model.stop_training = True

def build_model(input_shape, num_classes):
    try:
        print("Building 3D CNN model...")
        model = Sequential()

        # First convolutional block
        model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same',
                         input_shape=input_shape, kernel_regularizer=l2(0.0001)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

        # Second convolutional block
        model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=l2(0.0001)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))

        # Third convolutional block
        model.add(Conv3D(filters=128, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=l2(0.0001)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        # Fourth convolutional block
        model.add(Conv3D(filters=256, kernel_size=(3, 3, 3), padding='same', kernel_regularizer=l2(0.0001)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

        # Flatten and dense layers
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu', kernel_regularizer=l2(0.0001)))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.0001)))

        # Compile the model with Adam optimizer
        adam = Adam(learning_rate=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        print("Model built successfully.")
        return model
    except Exception as e:
        print(f"Error building model: {e}")
        raise

# Updated training section in the main function
def main():
    try:
        # Load Remote Sensing data and spectral profiles library
        RS_data, Map_Info, Coordinate_System_String = load_RS_data()  # Get map info and coordinate string
        mask = load_mask()
        spectral_data, labels, class_labels = load_all_spectral_libraries()
        RS_data_masked, valid_pixels = apply_mask_to_data(RS_data, mask)
        RS_data_reshaped = reshape_data_for_cnn(RS_data_masked, name="RS Data")
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(spectral_data, labels, test_size=0.15, random_state=42)

        # Reshape data to include all bands
        X_train = reshape_data_for_cnn(X_train, name="X_train")
        X_test = reshape_data_for_cnn(X_test, name="X_test")

        # Determine the number of classes dynamically
        num_classes = len(class_labels)

        # Convert labels to one-hot encoding
        y_train = to_categorical(y_train, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)

        # Build and train the model
        model = build_model(RS_data_reshaped.shape[1:], num_classes=num_classes)

        print("Training model...")

        metrics_callback = MetricsCallback(validation_data=(X_test, y_test))

        # ****************justify the accuracy_threshold, loss_threshold***************
        custom_early_stopping = CustomEarlyStopping(accuracy_threshold=0.993, loss_threshold=0.11)
        history = model.fit(
            X_train, y_train,
            batch_size=64, epochs=300,  # Set epochs to a large number to stop the callback
            verbose=1, validation_data=(X_test, y_test),
            callbacks=[metrics_callback, custom_early_stopping]
        )

        # Plot accuracy over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.show()

        # Plot loss over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.show()

        # Generate final classification report
        print("Generating final classification report...")
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)

        # Use class_labels dynamically for the target names
        report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
        print("\nFinal Classification Report:\n")
        print(report)

        # Confusion matrix
        print("Generating confusion matrix...")
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true_classes, y_pred_classes)
        print("Confusion Matrix:")
        print(cm)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.show()

        # Classify Remote Sensing image
        print("Classifying Remote Sensing image...")
        original_shape = RS_data.shape[:2]  # Save the original spatial dimensions
        num_classes = len(class_labels)  # Define the number of classes

        # Flatten valid_pixels
        valid_pixels = mask.flatten() == 1  # Ensure valid_pixels is a 1D boolean array
        print(f"Valid pixels count: {valid_pixels.sum()}")  # Check alignment with RS_predictions.shape[0]

        # Generate predictions
        RS_predictions = model.predict(RS_data_reshaped)
        print(f"Model predictions shape: {RS_predictions.shape}")  # Should be (valid_pixel_count, num_classes)

        # Add "Unclassified" to the class labels
        class_labels_with_unclassified = ["Unclassified"] + class_labels

        # Update the classified image to include unclassified regions
        classified_image = np.zeros(original_shape, dtype=int)
        classified_image_flat = classified_image.flatten()
        classified_image_flat[valid_pixels] = np.argmax(RS_predictions,
                                                        axis=1) + 1  # Offset by 1 to reserve 0 for "Unclassified"
        classified_image = classified_image_flat.reshape(original_shape)

        # Display classified image
        plt.figure(figsize=(10, 8))
        cmap = plt.cm.get_cmap('tab20', len(class_labels_with_unclassified))  # Ensure the colormap covers all classes
        im = plt.imshow(classified_image, cmap=cmap, interpolation='none')
        plt.title("Classified Image")

        # Add colorbar with class labels including "Unclassified"
        cbar = plt.colorbar(im, ticks=np.arange(len(class_labels_with_unclassified)))
        cbar.ax.set_yticklabels(class_labels_with_unclassified)  # Set class labels including "Unclassified"
        cbar.set_label('Class')
        plt.show()
    except Exception as e:
        print(f"An error occurred during processing: {e}")

    # Create abundance maps
    try:
        # Create abundance maps
        abundance_maps = np.zeros((original_shape[0], original_shape[1], num_classes))

        # Flatten valid_pixels to match predictions
        valid_pixels_flat = mask.reshape(-1) == 1  # Ensure valid_pixels is derived directly from the mask
        print(f"Valid pixels count: {valid_pixels_flat.sum()}")

        # Check alignment between predictions and valid pixels
        if RS_predictions.shape[0] != valid_pixels_flat.sum():
            raise ValueError("Mismatch between the number of predictions and the valid pixels count.")

        # Map predictions to the abundance maps
        abundance_maps_flat = abundance_maps.reshape(-1, num_classes)
        abundance_maps_flat[valid_pixels_flat] = RS_predictions
        abundance_maps = abundance_maps_flat.reshape(original_shape[0], original_shape[1], num_classes)

        print(f"Abundance maps shape: {abundance_maps.shape}")
    except Exception as e:
        print(f"Error mapping predictions to abundance maps: {e}")
        raise

    # Display abundance maps for each class
    fig, axes = plt.subplots(3, 3, figsize=(8, 6))  # Adjust nrows and ncols for your layout
    for i, ax in enumerate(axes.flatten()):
        if i < len(class_labels):  # Avoid indexing error if there are fewer classes than subplots
            ax.imshow(abundance_maps[:, :, i], cmap='viridis')
            ax.set_title(f'{class_labels[i]}')  # Dynamically use the class label
            ax.axis('off')
        else:
            ax.axis('off')  # Hide unused subplot
    plt.tight_layout()
    plt.show()

    # Save abundance maps
    output_dir = r"C:\\1urmia\\5clas"
    output_path = os.path.join(output_dir, "4 classified")
    os.makedirs(output_dir, exist_ok=True)
    # Set metadata for the ENVI file

    metadata = {
        'description': 'Abundance maps for mineral classification',
        'bands': len(class_labels),
        'data type': 4,  # float32
        'interleave': 'bsq',
        'byte order': 0,
        'map info': Map_Info,
        'coordinate system string': Coordinate_System_String,
        'band names': class_labels
    }

    envi.save_image(output_path + ".hdr", abundance_maps, dtype=np.float32, force=True, interleave='bsq',
                    metadata=metadata)
    print(f"Abundance maps saved at {output_path}.hdr")

if __name__ == "__main__":
    main()
