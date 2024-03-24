# Benjamin E Hogan
# 01/29/2024
# --- Testing and Obtaining Results from Neural Network --- # 


# -------- Modules --------- #
import tensorflow as tf
import numpy as np 
import os
import gwpy 
from tqdm import tqdm
from random import randint
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import time
import shutil
import re
import cv2
from neuralplot import ModelPlot


# -------- Global Vairables --------- #
BASE_DIR = os.getcwd()

# Directory paths
directories = {
    "Pre-normalized": os.path.join(BASE_DIR, 'data', 'Pre-normalized_Files'),
    "normalized": os.path.join(BASE_DIR, 'data', 'Normalized_Files'),
    "train": os.path.join(BASE_DIR, 'data', 'data_set', 'train'),
    "val": os.path.join(BASE_DIR, 'data', 'data_set', 'val'),
    "test": os.path.join(BASE_DIR, 'data', 'data_set', 'test'),
    "models": os.path.join(BASE_DIR, "data", "Saved Models"),
    "data": os.path.join(BASE_DIR, "data", "Strain data"),
    "nan_files": os.path.join(BASE_DIR, "data", "NaN_Files"),
    "raw_data": os.path.join(BASE_DIR, "data", "Raw URL Data"),
    "spectrograms": os.path.join(BASE_DIR, "data", "Spectrograms"),
    "Anomalies": os.path.join(BASE_DIR, "data", "Anomalies")
}


# ------- Helper Functions -------- #

def load_data(directory, batch_size=24):
    print(f"Listing files in {directory}...")
    file_paths = [os.path.join(directory, f) for f in tqdm(os.listdir(directory), desc="Listing Files") if f.endswith('.npy')]

    print(f"Loading data from {directory}...")
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    def load_file(file_path):
        data = np.load(file_path.numpy())
        data = np.expand_dims(data, axis=-1)  # Add a channel dimension
        return data, data

    dataset = dataset.map(lambda x: tf.py_function(load_file, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    print(f"Batching and prefetching...")
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    print("All data loaded and batched")

    return dataset

def display_reconstructions(model, dataset, n=10):
    # Shuffle the dataset and take 1 batch
    for test_images, _ in dataset.shuffle(1000).take(1):
        decoded_images = model.predict(test_images)

        # Set up the figure for plotting original and reconstructed data
        fig, axes = plt.subplots(2, n, figsize=(20, 4))
        time = np.linspace(0, 64, 262144)  # 64 seconds, 4096Hz

        # Find the global min and max to set consistent y-axis limits
        global_min = min(test_images.numpy().min(), decoded_images.min())
        global_max = max(test_images.numpy().max(), decoded_images.max())

        for i in range(n):
            # Ensure we do not go out of bounds if less than n examples are in the batch
            if i >= test_images.shape[0]:
                break

            # Original Strain Data
            ax = axes[0, i]
            ax.plot(time, test_images[i].numpy().flatten(), color='black')
            ax.set_title("Original")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Strain')
            ax.set_ylim(global_min, global_max)

            # Reconstructed Strain Data
            ax = axes[1, i]
            ax.plot(time, decoded_images[i].flatten(), color='red')
            ax.set_title("Reconstructed")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Strain')
            ax.set_ylim(global_min, global_max)

        plt.tight_layout()
        plt.show()

def create_directories(directory_paths):
    for path in directory_paths.values():
        os.makedirs(path, exist_ok=True)

# -------- Neural Network Functions -------- #

def load_latest_checkpoint(model, checkpoint_dir):
    ''' Grabs the newest trained model from models folder '''
    checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Loading weights from {latest_checkpoint}")
        model.load_weights(latest_checkpoint)
        
        
        # Extract the last epoch number from the filename using regex
        epoch_num_match = re.search(r"_(\d+).h5$", latest_checkpoint)
        last_epoch = int(epoch_num_match.group(1)) if epoch_num_match else 0
        print(f'Loaded lasted model at epoch {last_epoch}')
        return last_epoch
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0
  
def create_autoencoder(input_shape):
    input_img = Input(shape=input_shape)  # input_shape should be (steps, 1) for 1D arrays
    x = Conv1D(16, 3, activation='relu', padding='same')(input_img)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(8, 3, activation='relu', padding='same')(x)
    encoded = MaxPooling1D(2, padding='same')(x)
    
    x = Conv1D(8, 3, activation='relu', padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=[tf.keras.metrics.MeanSquaredError(), 
                                 tf.keras.metrics.MeanAbsoluteError()])

    return autoencoder

def evaluate_model(model, test_dataset):
    print("Evaluating model on test data...")
    test_loss, test_mse, test_mae = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}, Test MSE: {test_mse}, Test MAE: {test_mae}")
    return test_loss, test_mse, test_mae

def flag_anomalies(model, dataset, n=10, threshold=None):
    reconstruction_errors = []
    anomalies = []
    original_images = []
    reconstruction_image_paths = []  # For saving paths of reconstructed images

    # Iterate over the entire dataset and collect reconstruction errors
    for test_images, _ in dataset:
        decoded_images = model.predict(test_images)
        mse = np.mean(np.square(decoded_images - test_images), axis=(1, 2, 3))
        reconstruction_errors.append(mse)

        # Collect original images for later inspection
        original_images.extend(test_images.numpy())

        # Save reconstructed images and collect their paths
        for i, decoded_image in enumerate(decoded_images):
            reconstructed_image_path = os.path.join(directories["Reconstructions"], f"reconstructed_{i}_{int(time.time())}.png")
            matplotlib.image.imsave(reconstructed_image_path, decoded_image.squeeze(), cmap='viridis')
            reconstruction_image_paths.append(reconstructed_image_path)

    reconstruction_errors = np.concatenate(reconstruction_errors)
    original_images = np.array(original_images)

    # Determine the threshold if not provided
    if threshold is None:
        threshold = np.mean(reconstruction_errors) + np.std(reconstruction_errors)  # Off by a single standard deviation

    # Flag anomalies based on the threshold
    anomalies = reconstruction_errors > threshold

    # Display anomalies
    print(f"Threshold: {threshold}")
    print(f"Detected {np.sum(anomalies)} anomalies out of {len(reconstruction_errors)} samples.")

    # Get indices of anomalies
    anomaly_indices = np.where(anomalies)[0]
    print("Indices of detected anomalies:", anomaly_indices)

    anomaly_image_paths = []  # For saving paths of anomaly images
    # Save the anomalies
    for idx in anomaly_indices:
        # Using matplotlib to save as PNG
        anomaly_image_path = os.path.join(directories["Anomalies"], f"anomaly_{idx}_{int(time.time())}.png")
        matplotlib.image.imsave(anomaly_image_path, original_images[idx].squeeze(), cmap='viridis')
        anomaly_image_paths.append(anomaly_image_path)

    print(f"Anomalies saved in {directories['Anomalies']}")

    return anomalies, reconstruction_errors, anomaly_image_paths, reconstruction_image_paths

# Function to generate Excel report
def generate_excel_report(anomalies, anomaly_image_paths, reconstruction_image_paths):
    data = {
        "Original Image Path": anomaly_image_paths,
        "Reconstructed Image Path": reconstruction_image_paths,
        "Is Anomalous": anomalies
    }
    df = pd.DataFrame(data)
    excel_path = os.path.join(BASE_DIR, "anomaly_report.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Excel report saved at {excel_path}")

# -------- Define and run __MAIN__ ------ # 

def main():
    
    # --- Ensure Folder Exist --- #
    create_directories(directories)
    
    # ---- Load data ----- #
    test_data = load_data(directories["test"])
    train_data = load_data(directories["train"])

    # --- Load Model ---- #
    input_shape = (262144, 1) 
    autoencoder = create_autoencoder(input_shape)
    ModelPlot(model=autoencoder, grid=False, connection=True, linewidth=0.1)
    last_epoch = autoencoder.load_weights('/Volumes/Research Thesis (BH 2023)/Classifying-Normal-LIGO-Instrument-/data/Saved Models/best_model_17.h5')#load_latest_checkpoint(autoencoder, directories['models'])  )

    # --- Evaluate Model ---- #
    evaluate_model(autoencoder, test_data)
    display_reconstructions(autoencoder, test_data, n=10)
    
    # --- Flag Anomalies --- #
    anomalies, reconstruction_errors = flag_anomalies(autoencoder, test_data)

if __name__ == "__main__":
    main()