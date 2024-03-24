
# ------------------------------------------------------------ #
#                       Nerual Network                         #
# ------------------------------------------------------------ #

# -------------------- Modules ------------------- #
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm
import random
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import datetime
import subprocess
import webbrowser
import time
import re

# --------------- Global Vairables --------------- # 
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

#-------------- Helper Functions ------------- # 

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")


def load_data(directory, batch_size=24):
    print(f"Listing files in {directory}...")
    file_paths = [os.path.join(directory, f) for f in tqdm(os.listdir(directory), desc="Listing Files") if f.endswith('.npy')]

    print(f"Loading data from {directory}...")
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    def load_file(file_path):
        data = np.load(file_path.numpy())
        data = np.expand_dims(data, axis=-1)  # Add a channels dimension, reshaping to (262144, 1)
        return data, data

    dataset = dataset.map(lambda x: tf.py_function(load_file, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    print(f"Batching and prefetching...")
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    print("All data loaded and batched")

    num_samples = len(file_paths)
    steps = np.ceil(num_samples / batch_size)
    return dataset, steps

def load_test_data(directory, batch_size=24):
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

def check_array_type(file_path):
    
    # Load the file
    data = np.load(file_path)

    # Print the type of the array
    print(f"The type of the array in {file_path} is {type(data)}")
    print(f"The shape of the array is {data.shape}")
    print(f"The data type of the array elements is {data.dtype}")

def select_random_file(directory):
    # List all numpy files in the directory
    npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]

    # Check if there are any files
    if not npy_files:
        print("No .npy files found in the directory.")
        return None

    # Randomly select a file
    selected_file = random.choice(npy_files)
    selected_file_path = os.path.join(directory, selected_file)

    # Return the path of the selected file
    return selected_file_path


class TqdmCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epochs_bar = tqdm(total=self.params['steps'], position=0, leave=True)
        self.epochs_bar.set_description(f'Epoch {epoch + 1}/{self.params["epochs"]}')

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_bar.n = self.params['steps']
        self.epochs_bar.update(0)
        self.epochs_bar.close()

    def on_batch_end(self, batch, logs=None):
        self.epochs_bar.set_postfix(logs)
        self.epochs_bar.update(1)
        
# ------------ Neural Network Functions ------------ # 

from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D

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

def launch_tensorboard(log_dir):
    # Launch TensorBoard as a subprocess
    tensorboard_process = subprocess.Popen(['tensorboard', '--logdir', log_dir])

    # Open the TensorBoard URL in the web browser
    tensorboard_url = 'http://localhost:6006'
    print(f"TensorBoard is running at {tensorboard_url}")
    webbrowser.open(tensorboard_url)

    return tensorboard_process

def load_latest_checkpoint(model, checkpoint_dir):
    checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Loading weights from {latest_checkpoint}")
        model.load_weights(latest_checkpoint)
        
        # Extract the last epoch number from the filename using regex
        epoch_num_match = re.search(r"_(\d+).h5$", latest_checkpoint)
        last_epoch = int(epoch_num_match.group(1)) if epoch_num_match else 0
        return last_epoch
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0

class TensorBoardLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, model):
        super(TensorBoardLoggingCallback, self).__init__()
        self.log_dir = log_dir
        self.model = model
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
            with self.file_writer.as_default():
                if logs is not None:
                    tf.summary.scalar('loss', logs['loss'], step=epoch)
                    tf.summary.scalar('val_loss', logs['val_loss'], step=epoch)
                    if 'accuracy' in logs:
                        tf.summary.scalar('accuracy', logs['accuracy'], step=epoch)
                    if 'val_accuracy' in logs:
                        tf.summary.scalar('val_accuracy', logs['val_accuracy'], step=epoch)
                # Log Histogram Summaries for Weights and Biases
                for layer in self.model.layers:
                    weights = layer.get_weights()
                    if weights:
                        tf.summary.histogram(layer.name + '_weights', weights[0], step=epoch)
                        if len(weights) > 1:
                            tf.summary.histogram(layer.name + '_biases', weights[1], step=epoch)
                self.file_writer.flush()

# ---------------- Running Main -------------------- #
def main():
    
    # ----- Ensuring Files Exist ----- # 
    for dir_path in directories.values():
        create_directory_if_not_exists(dir_path)
    
    # ------ Loading Data Sets ------ # 
    train_dataset, train_steps = load_data(directories['train'], batch_size=24)
    val_dataset, val_steps = load_data(directories['val'], batch_size=24)
    test_dataset = load_test_data(directories['test'])
    
        # After loading datasets
    print("Train Dataset:")
    for x, y in train_dataset.take(1):
        print("x shape:", x.shape, "y shape:", y.shape)

    print("Validation Dataset:")
    for x, y in val_dataset.take(1):
        print("x shape:", x.shape, "y shape:", y.shape)

    print("Test Dataset:")
    for x, y in test_dataset.take(1):
        print("x shape:", x.shape, "y shape:", y.shape)
    
    # ---- checking array type and other analytics --- #
    random_file_path = select_random_file(directories['train'])
    if random_file_path:
        print(f"Randomly selected file: {random_file_path}")
        check_array_type(random_file_path)
        
    # ----- Auto Encoder Creation ----- # 
    # Define the input shape based on your data
    input_shape = (262144, 1)  # Assuming grayscale images, hence the 1 in the end

    autoencoder = create_autoencoder(input_shape)
    autoencoder.summary()
    
    # Load the latest checkpoint if exists
    last_epoch = load_latest_checkpoint(autoencoder, directories['models'])


    # 1. TensorBoard Summary Writer Setup
    log_dir = os.path.join(directories['models'], "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(log_dir)

    # 2. TensorBoard Callback for Visualization
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq='epoch')

    
    # Launch TensorBoard
    tensorboard_process = launch_tensorboard(log_dir)
    time.sleep(10)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    best_model_checkpoint = ModelCheckpoint(filepath='best_model.h5', save_best_only=True, monitor='val_loss')

    # Model Checkpoint Callback to Save the Best Model
    checkpoint_path = os.path.join(directories['models'], "best_model.h5")
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(directories['models'], "best_model_{epoch:02d}.h5"),
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        save_freq='epoch')


    ## ----- Auto Encoder Training ----- #

    # Then, in your main function, instantiate the TensorBoardLoggingCallback
    tensorboard_logging_callback = TensorBoardLoggingCallback(log_dir, autoencoder)

    history = autoencoder.fit(
        train_dataset,
        epochs=100,
        validation_data=val_dataset,
        callbacks=[tensorboard_callback, tensorboard_logging_callback, early_stopping, checkpoint_callback],
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        initial_epoch=last_epoch
    )
    # Optional: Visualizing Training Progress
    # This part is optional and can be used to plot the training and validation loss
    try:
        import matplotlib.pyplot as plt
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()
    except ImportError:
        print("matplotlib is not installed.")

    # ---- Optional: Evaluate on Test Set ---- #
    test_loss = autoencoder.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}")

if __name__ == "__main__":
    main()
