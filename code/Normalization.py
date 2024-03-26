import os
import shutil
import numpy as np
from gwpy.timeseries import TimeSeries # Used for Timeseries Manipulation (Not Used atm)
from tqdm import tqdm # Commandline progress meter
import concurrent.futures # Allows for parallel execution of task (Threading)

# ------------- Global Variables -------------- #
BASE_DIR = os.getcwd() #Gets current working directory

#Used for normalization between 0 and 1 for better, more efficent training
global_min = np.inf 
global_max = -np.inf

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
    "spectrograms": os.path.join(BASE_DIR, "data", "Spectrograms"),
    "Anomalies": os.path.join(BASE_DIR, "data", "Anomalies")
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

def contains_nan(array):
    """
    Checks if a given NumPy array contains any NaN values.
    
    Parameters:
    - array (np.array): The array to check for NaN values.
    
    Returns:
    - bool: True if there are any NaN values, False otherwise.
    """
    return np.any(np.isnan(array))

def update_global_min_max(directory):
    """
    Updates the global minimum and maximum values based on the data in .npy files within a given directory.
    
    Parameters:
    - directory (str): The directory to search for .npy files.
    """
    # Accesses the global variables to update them
    global global_min, global_max
    #List all .npy (Binary) files in the given directory
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]

    for filename in files:
        file_path = os.path.join(directory, filename)
        data = np.load(file_path)
        # Update the global min and max with the local min and max of the current file.
        local_min = np.min(data)
        local_max = np.max(data)
        global_min = min(global_min, local_min)
        global_max = max(global_max, local_max)

def process_file(filename):
    """
    Processes a single .npy file from the 'data' directory: loads the data, potentially transforms it,
    and saves the result in the 'Pre-normalized' directory. Updates the global min and max values based on the data.
    
    (Currently, This just finds global max and min and saves the file to the pre-processed directory)
    
    Parameters:
    - filename (str): The name of the file to process.
    """
    global global_min, global_max
    file_path = os.path.join(directories['data'], filename)
    data = np.load(file_path)
    
    # Create a Timeseries at 4096Hz and performs a Q-transform and performs a log on the absolute value of the data
    #data_ts = TimeSeries(data, sample_rate=4096)
    #data_q_transformed = data_ts.q_transform()
    #data_log = np.log(np.abs(data_q_transformed))
    
    # Saving the data to the right directory
    data_to_save = data.value if hasattr(data, 'value') else data
    q_transformed_file_path = os.path.join(directories['Pre-normalized'], 'Pre-normalized_'+filename)
    np.save(q_transformed_file_path, data_to_save)

    # Update global min and max based on the file's data
    local_min = np.min(data_to_save)
    local_max = np.max(data_to_save)
    global_min = min(global_min, local_min)
    global_max = max(global_max, local_max)

def normalize_data(filename, data_directory, normalized_directory, global_min, global_max):
    """
    Normalizes the data in a .npy file and saves the normalized data in the 'normalized' directory.
    
    Parameters:
    - filename (str): The name of the file to normalize.
    - data_directory (str): The directory containing the data to normalize.
    - normalized_directory (str): The directory where the normalized data should be saved.
    - global_min (float): The global minimum value for normalization.
    - global_max (float): The global maximum value for normalization.
    """
    file_path = os.path.join(data_directory, filename)
    data = np.load(file_path)

    normalized_data = (data - global_min) / (global_max - global_min)
    normalized_file_path = os.path.join(normalized_directory, 'normalized_'+filename)
    np.save(normalized_file_path, normalized_data)

def split_data(normalized_directory, train_directory, val_directory, test_directory, train_ratio=0.7, val_ratio=0.15):
    """
    Splits the data in 'normalized' directory into training, validation, and testing datasets.
    
    Parameters:
    - normalized_directory (str): The directory containing normalized data.
    - train_directory (str): The directory to save training data.
    - val_directory (str): The directory to save validation data.
    - test_directory (str): The directory to save testing data.
    - train_ratio (float): The ratio of data to be used for training.
    - val_ratio (float): The ratio of data to be used for validation.
    """
    # List all .npy files in the normalized directory
    all_files = [f for f in os.listdir(normalized_directory) if f.endswith('.npy')]
    np.random.shuffle(all_files) # Shuffle to ensure random distribution

    # Determine indices for splitting based on ratios in the parameters
    total_files = len(all_files)
    train_end = int(train_ratio * total_files)
    val_end = train_end + int(val_ratio * total_files)

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]
    
    # Copy files to their respective new directories
    def copy_files(files, destination):
        for f in tqdm(files, desc=f"Copying files to {os.path.basename(destination)}"):
            shutil.copy(os.path.join(normalized_directory, f), destination)

    copy_files(train_files, train_directory)
    copy_files(val_files, val_directory)
    copy_files(test_files, test_directory)

    print(f"Data split into train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)}")


# ------------- Running Main ---------------- # 
def main():
    '''
    Main routine to orchestrate:
    1. Directory setup
    2. Data Preprocessing
    3. Normalization
    4. Data set splitting
    '''
    
    # Create directories if they don't exist
    for dir_path in directories.values():
        create_directory_if_not_exists(dir_path)
    

    # Process files: File cleaning for NaN values
    for filename in tqdm(os.listdir(directories['data']), desc="Checking for NaN Files"):
        if filename.endswith('.npy'):
            file_path = os.path.join(directories['data'], filename)
            data = np.load(file_path)
            if contains_nan(data):
                new_location = os.path.join(directories['nan_files'], filename)
                os.rename(file_path, new_location)
                print(f"Moved {filename} to {directories['nan_files']} due to NaN values.")
   
    # Main data processing and normalization
    num_threads = 15 # Define the number of threads for parallel processing

    # Loading & preprocessing data with threading
    npy_files = [f for f in os.listdir(directories['data']) if f.endswith('.npy')]
    
    # Count the number of files in each directory
    strain_data_files = [f for f in os.listdir(directories['data']) if f.endswith('.npy')]
    q_transformed_files = [f for f in os.listdir(directories['Pre-normalized']) if f.endswith('.npy')]

    # Processing of all .npy (Binary) files
    # checks to see if the files have already been processed
    if len(strain_data_files) == len(q_transformed_files):
        print("All files have already been processed. Updating global min and max.")
        update_global_min_max(directories['Pre-normalized'])
        npy_files = q_transformed_files
    else:
        # Processing of files is completed here
        print("Processing new files.")
        npy_files = strain_data_files
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(tqdm(executor.map(process_file, npy_files), total=len(npy_files), desc="Processing Files"))

    npy_files = [f for f in os.listdir(directories['Pre-normalized']) if f.endswith('.npy')]
    # Continue with normalization
    for filename in tqdm(npy_files, desc="Normalizing Files"):
        normalize_data(filename, directories['Pre-normalized'], directories['normalized'], global_min, global_max)

    print("Normalization process completed.")
    
    # Splitting the data with tqdm monitoring
    split_data(directories['normalized'], directories['train'], directories['val'], directories['test'])


if __name__ == "__main__":
    main()
    
# This code block concludes the processing of files including their normalization and the splitting of the data into training, validation, and testing datasets. Here's a quick recap of what happens:

# 1. **Main Data Processing**: After ensuring all necessary directories are created, the script processes each `.npy` file. It checks for NaN values, processes files (currently, this just involves saving them without transformation), and updates global minimum and maximum values.

# 2. **Normalization**: Each pre-normalized file is then normalized based on the global minimum and maximum values calculated from the pre-normalized data. This normalization scales the data to a range between 0 and 1.

# 3. **Data Splitting**: Finally, the script splits the normalized data into training, validation, and testing datasets according to specified ratios. It then copies these files into their respective directories for use in machine learning models.

# 4. **Parallel Processing**: The script makes use of `concurrent.futures.ThreadPoolExecutor` for parallel processing of file processing tasks to speed up the execution.

# This script ensures a systematic approach to preparing data for machine learning models, from initial preprocessing and normalization to splitting the dataset into distinct sets for training and evaluation.
