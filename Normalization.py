import os
import shutil
import numpy as np
from gwpy.timeseries import TimeSeries
from tqdm import tqdm
import concurrent.futures

# ------------- Global Variables -------------- #
BASE_DIR = os.getcwd()
print(BASE_DIR)
global_min = np.inf
global_max = -np.inf

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
    "Anomalies": os.path.join(BASE_DIR, "data", "Anomalies")
}

#-------------- Helper Functions ------------- # 

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def contains_nan(array):
    return np.any(np.isnan(array))

def update_global_min_max(directory):
    global global_min, global_max
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]

    for filename in files:
        file_path = os.path.join(directory, filename)
        data = np.load(file_path)

        # Update the global min and max
        local_min = np.min(data)
        local_max = np.max(data)

        global_min = min(global_min, local_min)
        global_max = max(global_max, local_max)

def process_file(filename):
    global global_min, global_max
    # Define file paths
    file_path = os.path.join(directories['data'], filename)
    q_transformed_path = os.path.join(directories['Q-transformed'], filename)

    # Load and process the data
    data = np.load(file_path)
    data_ts = TimeSeries(data, sample_rate=4096)
    data_q_transformed = data_ts.q_transform()
    data_log = data_q_transformed

    # Convert the Spectrogram object to a NumPy array
    data_to_save = data_log.value if hasattr(data_log, 'value') else data_log

    # Save the Q-transformed data
    np.save(q_transformed_path, data_to_save)

    # Update global min and max
    local_min = np.min(data_to_save)
    local_max = np.max(data_to_save)

    # Thread-safe updates
    global_min = min(global_min, local_min)
    global_max = max(global_max, local_max)

def normalize_data(filename, data_directory, normalized_directory, global_min, global_max):
    file_path = os.path.join(data_directory, filename)
    data = np.load(file_path)

    normalized_data = (data - global_min) / (global_max - global_min)

    # Save the normalized data to the new directory
    normalized_file_path = os.path.join(normalized_directory, 'normalized_' + filename)
    np.save(normalized_file_path, normalized_data)

def split_data(normalized_directory, train_directory, val_directory, test_directory, train_ratio=0.7, val_ratio=0.15):
    # Get all file names
    all_files = [f for f in os.listdir(normalized_directory) if f.endswith('.npy')]
    np.random.shuffle(all_files)  # Shuffle the files

    # Calculate split indices
    total_files = len(all_files)
    train_end = int(train_ratio * total_files)
    val_end = train_end + int(val_ratio * total_files)

    # Split files
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    # Function to copy files to respective directories with tqdm monitoring
    def copy_files(files, destination):
        for f in tqdm(files, desc=f"Copying files to {os.path.basename(destination)}"):
            shutil.copy(os.path.join(normalized_directory, f), destination)

    # Copy files to respective directories
    copy_files(train_files, train_directory)
    copy_files(val_files, val_directory)
    copy_files(test_files, test_directory)

    print(f"Data split into train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")


# ------------- Running Main ---------------- # 
def main():
    # Create directories
    for dir_path in directories.values():
        create_directory_if_not_exists(dir_path)
    

    # File cleaning for NaN values
    for filename in tqdm(os.listdir(directories['data']), desc="Checking for NaN Files"):
        if filename.endswith('.npy'):
            file_path = os.path.join(directories['data'], filename)
            data = np.load(file_path)

            if contains_nan(data):
                new_location = os.path.join(directories['nan_files'], filename)
                os.rename(file_path, new_location)
                print(f"Moved {filename} to {directories['nan_files']} due to NaN values.")

    # Define the number of threads
    num_threads = 20

    # Loading & preprocessing data with threading
    npy_files = [f for f in os.listdir(directories['data']) if f.endswith('.npy')]
    
    # Count the number of files in each directory
    strain_data_files = [f for f in os.listdir(directories['data']) if f.endswith('.npy')]
    q_transformed_files = [f for f in os.listdir(directories['Q-transformed']) if f.endswith('.npy')]

    if len(strain_data_files) == len(q_transformed_files):
        print("All files have already been processed. Updating global min and max.")
        update_global_min_max(directories['Q-transformed'])
        npy_files = q_transformed_files
    else:
        print("Processing new files.")
        npy_files = strain_data_files
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(process_file, npy_files), total=len(npy_files), desc="Processing Files"))


    # Continue with normalization
    for filename in tqdm(npy_files, desc="Normalizing Files"):
        normalize_data(filename, directories['Q-transformed'], directories['normalized'], global_min, global_max)

    print("Normalization process completed.")
    
    # Splitting the data with tqdm monitoring
    split_data(directories['normalized'], directories['train'], directories['val'], directories['test'])


if __name__ == "__main__":
    main()
