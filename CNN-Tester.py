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
    "Q-transformed": os.path.join(BASE_DIR, 'data', 'Q-transformed_Files'),
    "normalized": os.path.join(BASE_DIR, 'data', 'Normalized_Files'),
    "train": os.path.join(BASE_DIR, 'data', 'data_set', 'train'),
    "val": os.path.join(BASE_DIR, 'data', 'data_set', 'val'),
    "test": os.path.join(BASE_DIR, 'data', 'data_set', 'test'),
    "models": os.path.join(BASE_DIR, "data", "Saved Models"),
    "data": os.path.join(BASE_DIR, "data", "Strain data"),
    "nan_files": os.path.join(BASE_DIR, "data", "NaN_Files"),
    "raw_data": os.path.join(BASE_DIR, "data", "Raw URL Data"),
    "spectrograms": os.path.join(BASE_DIR, "data", "Spectrograms"),
    "Anomalies": os.path.join(BASE_DIR, "data", "Anomalies"),
    "Reconstructions": os.path.join(BASE_DIR, "data", "Reconstructions")  # New directory for saving reconstructed images
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
    # Take a single batch from the dataset
    for test_images, _ in dataset.take(1):
        decoded_images = model.predict(test_images)
        
        fig = plt.figure(figsize=(20, 8))  # Larger figure size
        gs = gridspec.GridSpec(2, n, hspace=0.4, wspace=0.05)  # Define a grid spec
        
        for i in range(n):
            # Original Image
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(test_images[i].numpy().squeeze(), cmap='viridis', aspect='auto')
            ax.set_title("Original", fontsize=10)
            ax.axis('off')

            # Reconstructed Image
            ax = fig.add_subplot(gs[1, i])
            ax.imshow(decoded_images[i].squeeze(), cmap='viridis', aspect='auto')
            ax.set_title("Reconstructed", fontsize=10)
            ax.axis('off')

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
    print('Begining Autoencoder Building')
    input_img = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    
    cropping = ((0, 0), (0, 1))  # Adjust as needed
    x = Cropping2D(cropping=cropping)(x)
    
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=[tf.keras.metrics.MeanSquaredError(), 
                                tf.keras.metrics.MeanAbsoluteError()])

    print('Autoencoder Building Complete')
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
    
    # Print the shape of a single batch of test data
    for images, _ in test_data.take(1):  # Take one batch from the dataset
        print("Shape of test data batch:", images.shape)

    # --- Load Model ---- #
    input_shape = (1000, 2571, 1)
    autoencoder = create_autoencoder(input_shape)
    ModelPlot(model=autoencoder, grid=False, connection=True, linewidth=0.1)
    last_epoch = autoencoder.load_weights('/Users/benjaminhogan/Code/Senior Thesis PT2/data/Saved Models/best_model_03.h5')#load_latest_checkpoint(autoencoder, directories['models'])

    # --- Evaluate and Display Reconstructions ---- #
    #evaluate_model(autoencoder, test_data)
    #display_reconstructions(autoencoder, test_data, n=10)
    
    # --- Flag Anomalies --- #
    anomalies, reconstruction_errors, anomaly_image_paths, reconstruction_image_paths = flag_anomalies(autoencoder, test_data)
    
    # --- Create Excel Report ---#
    generate_excel_report(anomalies, anomaly_image_paths, reconstruction_image_paths)

if __name__ == "__main__":
    main()