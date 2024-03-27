# Benjamin Hogan
# 03/25/24 (Last Updated)


# ------------------------------------------------------------ #
#                       Nerual Network                         #
# ------------------------------------------------------------ #

# -------------------- Modules ------------------- #
import os 
import tensorflow as tf # Core library for machine learning and neural networks
from tensorflow import keras # High level API for building and training models in Tensorflow
import numpy as np
from tqdm import tqdm # Commandline Progess bars
import random
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D # Layers for building neural networ modes
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback # Utils to add specific funtions to the training process 
from tensorflow.keras.models import Model # Keras model for creating and training neural networks
import matplotlib.pyplot as plt
import datetime
import subprocess
import webbrowser
import time
import re # Regular Expression Mathcing

# --------------- Global Vairables --------------- # 
BASE_DIR = os.getcwd() #Gets the current working directory

# Directory paths
directories = {
    #Creates a dictionary to store paths of various directories
    "Pre-normalized": os.path.join(BASE_DIR, 'data', 'Pre-normalized_Files'),
    "normalized": os.path.join(BASE_DIR, 'data', 'Normalized_Files'),
    "train": os.path.join(BASE_DIR, 'data', 'data_set', 'train'),
    "val": os.path.join(BASE_DIR, 'data', 'data_set', 'val'),
    "test": os.path.join(BASE_DIR, 'data', 'data_set', 'test'),
    "models": os.path.join(BASE_DIR, "data", "Saved Models"),
    "data": os.path.join(BASE_DIR, "data", "Strain data"),
    "nan_files": os.path.join(BASE_DIR, "data", "NaN_Files"),
    "raw_data": os.path.join(BASE_DIR, "data", "Raw URL Data"),
    "spectrograms": os.path.join(BASE_DIR, "data", "images", "Spectrograms"),
    "Anomalies": os.path.join(BASE_DIR, "data", "images", "Anomalies"),
    "Reconstructions": os.path.join(BASE_DIR, "data", "images", "Reconstructions"),
    "images": os.path.join(BASE_DIR, "data", "images"),
    "Comparison": os.path.join(BASE_DIR, "data", "images", "Comparisons"),
    "Reports": os.path.join(BASE_DIR, "Reports")
}

#-------------- Helper Functions ------------- # 

def create_directory_if_not_exists(directory):
    """
    Checks if a directory exists in the file system; if not, it creates it.
    
    Parameters:
    - directory (str): The path of the directory to check or create.
    """
    # Checks if directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def load_data(directory, batch_size=24):
    """
    Loads data from numpy (.npy) files within a specified directory, batches it, and returns a TensorFlow dataset.

    Args:
        directory (str): The path to the directory containing the .npy files.
        batch_size (int, optional): The size of the batches of data. Defaults to 24.

    Returns:
        tuple: A tuple containing the TensorFlow dataset and the number of steps per epoch.
    """
    
    print(f"Listing files in {directory}...")
    file_paths = [os.path.join(directory, f) for f in tqdm(os.listdir(directory), desc="Listing Files") if f.endswith('.npy')]

    print(f"Loading data from {directory}...")
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    def load_file(file_path):
        """
        Loads a numpy file and adds a channel dimension to its data.

        Args:
            file_path (Tensor): A TensorFlow tensor containing the file path.

        Returns:
            tuple: A tuple containing the loaded data and itself (for autoencoder purposes).
        """
        
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
    """
    Loads test data from numpy (.npy) files within a specified directory, batches it, and returns a TensorFlow dataset.
    Similar to `load_data` but tailored for test data loading.

    Args:
        directory (str): The path to the directory containing the .npy files.
        batch_size (int, optional): The size of the batches of data. Defaults to 24.

    Returns:
        tf.data.Dataset: The TensorFlow dataset containing the batched test data.
    """
    
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
    """
    Loads a numpy file and prints its array type, shape, and data type of its elements.
    
    This is mostly for Debugging purposes

    Args:
        file_path (str): The path to the numpy (.npy) file.
    """
    
    # Load the file
    data = np.load(file_path)

    # Print the type of the array
    print(f"The type of the array in {file_path} is {type(data)}")
    print(f"The shape of the array is {data.shape}")
    print(f"The data type of the array elements is {data.dtype}")

def select_random_file(directory):
    """
    Selects a random .npy file from a specified directory and returns its path.

    Args:
        directory (str): The path to the directory from which to select a random file.

    Returns:
        str or None: The path to the randomly selected file, or None if no .npy files are found.
    """
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
    """
    A custom TensorFlow callback to display training progress in a tqdm progress bar.

    Inherits from:
        Callback (tensorflow.keras.callbacks.Callback)
    """
    def on_epoch_begin(self, epoch, logs=None):
        """
        Initializes a tqdm progress bar at the beginning of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Currently unused parameter, logs from the training process.
        """
        self.epochs_bar = tqdm(total=self.params['steps'], position=0, leave=True)
        self.epochs_bar.set_description(f'Epoch {epoch + 1}/{self.params["epochs"]}')

    def on_epoch_end(self, epoch, logs=None):
        """
        Closes the tqdm progress bar at the end of each epoch.

        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Currently unused parameter, logs from the training process.
        """
        self.epochs_bar.n = self.params['steps']
        self.epochs_bar.update(0)
        self.epochs_bar.close()

    def on_batch_end(self, batch, logs=None):
        """
        Updates the tqdm progress bar at the end of each batch.

        Args:
            batch (int): The current batch number within the epoch.
            logs (dict, optional): The logs from the training process, used for updating the progress bar postfix.
        """
        self.epochs_bar.set_postfix(logs)
        self.epochs_bar.update(1)
        
def print_dataset_shapes(dataset, dataset_name):
    """
    Prints the shape of the first batch of a given dataset.

    Args:
        dataset (tf.data.Dataset): The TensorFlow dataset to print shapes for.
        dataset_name (str): A descriptive name for the dataset to include in the print statements.
    """
    print(f"{dataset_name} Dataset:")
    for x, y in dataset.take(1):
        print("x shape:", x.shape, "y shape:", y.shape)
        
# ------------ Neural Network Functions ------------ # 
def create_autoencoder(input_shape):
    """
    Constructs an autoencoder neural network for 1D signal data. Autoencoders are a type of artificial neural network 
    used to learn efficient codings of unlabeled data. The aim of an autoencoder is to learn a representation 
    (encoding) for a set of data, typically for the purpose of dimensionality reduction. This implementation focuses 
    on 1D signal data, such as time series or audio signals, utilizing 1D convolutional layers.

    The architecture of this autoencoder consists of two main parts: the encoder and the decoder. The encoder part 
    compresses the input into a smaller, dense representation, and the decoder part attempts to reconstruct the 
    input from this compressed representation.

    Args:
        input_shape (tuple): The shape of the input data, expected to be in the format (steps, 1) for 1D arrays. The 'steps' 
                             represents the dimensionality or length of the input signal, and '1' signifies that each step is 
                             a scalar (as opposed to a vector of features).

    Returns:
        Model: A compiled TensorFlow/Keras model of the autoencoder with the Adam optimizer and mean squared error loss.

    The model uses ReLU activations for the encoder and decoder intermediate layers to introduce non-linearity, 
    allowing it to learn more complex patterns in the data. The final layer uses a sigmoid activation function 
    to ensure the output values are in the range [0, 1], which is particularly useful if the input data has been 
    normalized to this range.

    This autoencoder can be used for various applications including but not limited to anomaly detection in time series, 
    noise reduction, and feature extraction.
    """
    
    # Input layer specifying the shape of input data
    input_img = Input(shape=input_shape)  
    
    # Encoder
    # Conv1D layer with 16 filters, kernel size of 3, using 'relu' activation and 'same' padding
    x = Conv1D(16, 3, activation='relu', padding='same')(input_img)
    # MaxPooling1D layer for downsampling the input by a factor of 2, using 'same' padding
    x = MaxPooling1D(2, padding='same')(x)
    # Another Conv1D layer, now with 8 filters, to further process the data
    x = Conv1D(8, 3, activation='relu', padding='same')(x)
    # Final MaxPooling1D in the encoder part for further downsampling
    encoded = MaxPooling1D(2, padding='same')(x)
    
    # Decoder
    # Conv1D layer that starts the decoding process, with 8 filters
    x = Conv1D(8, 3, activation='relu', padding='same')(encoded)
    # UpSampling1D layer to increase the dimensionality of the data back to its original size
    x = UpSampling1D(2)(x)
    # Conv1D layer with 16 filters to further refine the decoded signal
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    # Another UpSampling1D layer to ensure the output size matches the input size
    x = UpSampling1D(2)(x)
    
    # Output layer to reconstruct the input signal. Uses a sigmoid activation to ensure the output
    # values are between 0 and 1, assuming the input data was normalized to this range.
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)

    # Creation of the autoencoder model, specifying the input and output layers
    autoencoder = Model(input_img, decoded)
    # Compiling the model with Adam optimizer and mean squared error as the loss function.
    # Additionally, it tracks mean squared error and mean absolute error as metrics for evaluation.
    autoencoder.compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=[tf.keras.metrics.MeanSquaredError(), 
                                 tf.keras.metrics.MeanAbsoluteError()])

    return autoencoder

def launch_tensorboard(log_dir):
    """
    Launches TensorBoard in a subprocess and opens it in a web browser.

    Args:
        log_dir (str): The directory where TensorBoard logs are stored.

    Returns:
        subprocess.Popen: The subprocess in which TensorBoard is running.
    """
    # Launch TensorBoard as a subprocess
    tensorboard_process = subprocess.Popen(['tensorboard', '--logdir', log_dir])

    # Open the TensorBoard URL in the web browser
    tensorboard_url = 'http://localhost:6006'
    print(f"TensorBoard is running at {tensorboard_url}")
    webbrowser.open(tensorboard_url)

    return tensorboard_process

def load_latest_checkpoint(model, checkpoint_dir):
    """
    Loads the latest checkpoint (weights) from a specified directory into a model.

    Args:
        model (Model): The TensorFlow model into which to load the weights.
        checkpoint_dir (str): The directory from which to load the latest checkpoint.

    Returns:
        int: The epoch number of the last checkpoint or 0 if no checkpoint was found.
    """
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
    """
    A custom TensorFlow callback for logging training metrics to TensorBoard.

    Args:
        log_dir (str): The directory where TensorBoard logs should be written.
        model (Model): The TensorFlow model being trained.
    """

    def __init__(self, log_dir, model):
        super(TensorBoardLoggingCallback, self).__init__()
        self.log_dir = log_dir
        self.model = model
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
            """
            Logs custom metrics at the end of each epoch to TensorBoard.

            Args:
                epoch (int): The current epoch number.
                logs (dict): A dictionary containing the training and validation metrics.
            """
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
    """
    Main function to orchestrate data preparation, model training, and evaluation processes.

    This function performs the following steps:
    1. Ensures that necessary directories exist for storing datasets, models, and logs.
    2. Loads training, validation, and test datasets.
    3. Demonstrates how to access and print the shape of the datasets.
    4. Selects a random file from the training dataset for analysis.
    5. Creates an autoencoder model based on a predefined input shape.
    6. Attempts to load the latest model checkpoint to resume training from the last saved state.
    7. Sets up TensorBoard for training visualization.
    8. Launches TensorBoard in a web browser for real-time monitoring.
    9. Configures early stopping and model checkpointing callbacks for efficient training.
    10. Starts training the model with the training and validation datasets, utilizing the callbacks.
    """
    
    # ----- Ensuring Files Exist ----- # 
    # Ensure each directory needed in the process exists, creating them if necessary
    for dir_path in directories.values():
        create_directory_if_not_exists(dir_path)
    
    # ------ Loading Data Sets ------ # 
    # Load datasets for training, validation, and testing
    train_dataset, train_steps = load_data(directories['train'], batch_size=24)
    val_dataset, val_steps = load_data(directories['val'], batch_size=24)
    test_dataset = load_test_data(directories['test'])
    
    # After loading datasets, print the shapes of the first batch for each dataset
    #print_dataset_shapes(train_dataset, "Train")
    #print_dataset_shapes(val_dataset, "Validation")
    #print_dataset_shapes(test_dataset, "Test")
    
    # ---- checking array type and other analytics --- #
    # Select a random file for analysis
    random_file_path = select_random_file(directories['train'])
    if random_file_path:
        print(f"Randomly selected file: {random_file_path}")
        check_array_type(random_file_path)
        
    # ----- Auto Encoder Creation ----- # 
    # Define the input shape for the autoencoder model
    input_shape = (262144, 1)  # Example input shape, adjust according to your data

    # Create the autoencoder model
    autoencoder = create_autoencoder(input_shape)
    autoencoder.summary()
    
    # Load the latest checkpoint if exists to continue training from there
    last_epoch = load_latest_checkpoint(autoencoder, directories['models'])

    # 1. TensorBoard Summary Writer Setup
    # Create a directory for TensorBoard logs with a timestamp
    log_dir = os.path.join(directories['models'], "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    file_writer = tf.summary.create_file_writer(log_dir)

    # 2. TensorBoard Callback for Visualization
    # Setup TensorBoard callback for training visualization
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq='epoch')

    # Launch TensorBoard
    # Note: This will open TensorBoard in a new browser tab
    tensorboard_process = launch_tensorboard(log_dir)
    # Wait for TensorBoard to launch
    time.sleep(3)
    
    # Setup Early Stopping and Model Checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    best_model_checkpoint = ModelCheckpoint(filepath='best_model.h5', save_best_only=True, monitor='val_loss')

    # Model Checkpoint Callback to Save the Best Model at each epoch
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
        epochs=25,
        validation_data=val_dataset,
        callbacks=[tensorboard_callback, tensorboard_logging_callback, early_stopping, checkpoint_callback],
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        initial_epoch=last_epoch
    )
    # # Optional: Visualizing Training Progress
    # # This part is optional and can be used to plot the training and validation loss
    # try:
    #     import matplotlib.pyplot as plt
    #     plt.plot(history.history['loss'], label='Training Loss')
    #     plt.plot(history.history['val_loss'], label='Validation Loss')
    #     plt.title('Training and Validation Loss')
    #     plt.legend()
    #     plt.show()
    # except ImportError:
    #     print("matplotlib is not installed.")

    # # ---- Optional: Evaluate on Test Set ---- #
    # test_loss = autoencoder.evaluate(test_dataset)
    # print(f"Test Loss: {test_loss}")

if __name__ == "__main__":
    main()
