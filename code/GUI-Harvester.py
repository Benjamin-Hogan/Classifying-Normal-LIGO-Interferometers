#---- Import Section -----
import os
import time
import random
import logging
import platform
import subprocess
from time import perf_counter
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock, Event
import tkinter as tk
from tkinter import ttk, PhotoImage, messagebox, simpledialog, Text, filedialog
import numpy as np
import requests
import matplotlib.pyplot as plt
import pygame
from PIL import Image, ImageTk
from scipy.io.wavfile import write as wav_write, read
from gwosc.locate import get_urls
from gwpy.timeseries import TimeSeries
import sounddevice as sd

# ----- Global Variables -----
BASE_DIR = os.getcwd()
cancel_operation = False
operation_complete = False
selected_detector = 'H1'  # Default
num_workers_selected = 20  # Default
spectrogram_folder = None
raw_data_folder = None
options_dialog = None
audio_stream = None
is_playing = False
audio_data = None
audio_rate = None
data_idx = 0
audio_lock = Lock()
# Global variable for start_time
start_time = None
cancel_event = Event()


notebook_path = os.getcwd()  # Get the current directory
BASE_DIR = notebook_path

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

def set_cancel_flag():
    cancel_event.set()

def open_folder(path):
    if platform.system() == 'Darwin':  # macOS
        subprocess.Popen(["open", path])
    elif platform.system() == 'Windows':
        os.startfile(path)
    else:  # Linux and other UNIX
        subprocess.Popen(["xdg-open", path])

def ensure_folder_exists(folder_name):
    full_path = os.path.join(BASE_DIR, folder_name)
    if not os.path.exists(full_path):
        #log_message(f"Creating folder: {full_path}\n", log_panel)
        os.makedirs(full_path)
    return full_path

def random_gps_time():
    start_gps_time = 946728000
    end_gps_time = 1261871999
    return random.randint(start_gps_time, end_gps_time)

def update_progress_bar():
    progress_bar["value"] += 1
    root.update_idletasks()

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")

def update_count(spectrogram_count_label, raw_data_count_label, spectrogram_folder, raw_data_folder):
    while True:
        try:
            num_spectrograms = len([f for f in os.listdir(spectrogram_folder) if os.path.isfile(os.path.join(spectrogram_folder, f))]) 
            num_raw_files = len([f for f in os.listdir(raw_data_folder) if os.path.isfile(os.path.join(raw_data_folder, f))]) 
            spectrogram_count_label.config(text=f"Number of Strain Files: {num_spectrograms}")
            raw_data_count_label.config(text=f"Number of Raw Data Files: {num_raw_files}")
        except Exception as e:
            print(f"An error occurred: {e}")
        time.sleep(1)

def download_file(gps_time, detector, progress_bar):
    log_message(f"Starting download_file function for GPS time {gps_time}.\n", log_panel)
    start_time = perf_counter()
    while True:
        if cancel_event.is_set():
            return
        try:
            print(f"Thread for GPS time {gps_time} started")
            url_found = False
            while not url_found:
                            # Check if cancel_event has been set after a potentially long-running operation
                if cancel_event.is_set():
                    return
                try:
                    url = get_urls(detector, gps_time, gps_time)[-1]
                    url_found = True
                except ValueError:
                    print(f"No URL found for GPS time {gps_time}. Trying a different GPS time.")
                    gps_time = random_gps_time()

            print(f"Downloading: {url}")
            log_message(f"Downloading: {url}\n", log_panel)

            filename = os.path.basename(url)
            filepath = filepath = os.path.join(directories['raw_data'], filename)
            with open(filepath, 'wb') as strainfile:
                straindata = requests.get(url)
                strainfile.write(straindata.content)
                
            # Check if cancel_event has been set after a potentially long-running operation
            if cancel_event.is_set():
                return
                
            strain = TimeSeries.read(filepath, format='hdf5.gwosc')
            start_time = strain.t0.value  # This will give you the start time in GPS format
            end_time = strain.t0.value + strain.duration.value  # This will give you the end time in GPS format
            center_time = int(start_time + (end_time - start_time) / 2)
            strain = strain.crop(center_time - 32, center_time + 32)
            
            strain_path = directories["data"]
            strain_data_path = os.path.join(strain_path , f'strain around {center_time}.npy')
            np.save(strain_data_path, strain.value)
            

            print(f"Saving plot for GPS time {center_time}")
            print(f"Thread for GPS time {gps_time} completed")
            end_time = perf_counter()
            elapsed_time = end_time - start_time
            root.after(0, update_progress_bar)
            log_message(f"Successfully downloaded and processed data for GPS time {gps_time}.\n", log_panel)
            os.remove(filepath)

            break
        except Exception as e:
            log_message(f"An error occurred for GPS time {gps_time}: {e}. Retrying with a new GPS time.", log_panel)
            gps_time = random_gps_time()
            os.remove(filepath)
            

def set_cancel_flag():
    global cancel_operation
    cancel_event.set()  # This line sets the cancel_event
    cancel_operation = True  # This line sets the cancel_operation global variable

    
def update_status_label(completed_tasks, total_num_tasks):
    status_label.config(text=f"{completed_tasks}/{total_num_tasks} tasks completed.")
    root.update_idletasks()

def update_total_elapsed_time(start_time):
    current_time = time.time()
    elapsed_time = current_time - start_time
    total_time_label.config(text=f"Total Elapsed Time: {elapsed_time:.2f} seconds")
    root.after(100, update_total_elapsed_time, start_time)  # Update every 100 milliseconds

def update_elapsed_time(start_time, total_time_label):
    global operation_complete  # Use the global flag
    if not operation_complete:  # Check the flag before updating
        current_time = time.time()
        elapsed_time = current_time - start_time
        total_time_label.config(text=f"Total Elapsed Time: {elapsed_time:.2f} seconds")
        root.after(10, update_elapsed_time, start_time, total_time_label) 

def update_average_time(total_elapsed_time, completed_tasks):
    global operation_complete  # Use the global flag
    if not operation_complete:  # Check the flag before updating
        if completed_tasks > 0:
            average_time = total_elapsed_time / completed_tasks
            average_time_label.config(text=f"Average Time for Each Spectrogram: {average_time:.2f} seconds")
        root.after(100, update_average_time, total_elapsed_time, completed_tasks)

def increment_completed_tasks():
    global completed_tasks
    with completed_tasks_lock:
        completed_tasks += 1
    
def show_options_dialog():
    global options_dialog, selected_detector, num_workers_selected
    if options_dialog:
        options_dialog.destroy()
        options_dialog = None
        return

    options_dialog = tk.Toplevel(root)
    options_dialog.title("Configuration Settings")
    
    detector_label = tk.Label(options_dialog, text="Detector:")
    detector_label.pack()
    
    detector_entry = tk.Entry(options_dialog)
    detector_entry.insert(0, selected_detector)
    detector_entry.pack()
    
    threads_label = tk.Label(options_dialog, text="Number of Threads:")
    threads_label.pack()
    
    threads_entry = tk.Entry(options_dialog)
    threads_entry.insert(0, str(num_workers_selected))
    threads_entry.pack()

    def apply_and_close():
        global selected_detector, num_workers_selected
        selected_detector = detector_entry.get()
        num_workers_selected = int(threads_entry.get())
        options_dialog.destroy()

    apply_button = tk.Button(options_dialog, text="Apply", command=apply_and_close)
    apply_button.pack()

    options_dialog.protocol("WM_DELETE_WINDOW", lambda: options_dialog.destroy())

# Function to show a help dialog
def show_help_dialog():
    help_text = """This application allows you to download and analyze gravitational wave data to generate spectrograms.
    - Enter the total number of tasks to specify how many spectrograms you want to generate.
    - Click 'Execute Script' to start the operation.
    - Use 'Cancel Operation' to stop the ongoing process."""
    messagebox.showinfo("Help", help_text)

def update_progress_bar_and_percentage(completed_tasks, total_num_tasks):
    progress_bar["value"] = completed_tasks
    percentage = (completed_tasks / total_num_tasks) * 100
    percentage_label.config(text=f"{percentage:.2f}%")
    root.after(100, update_progress_bar_and_percentage, completed_tasks, total_num_tasks)

# Global lock for updating completed_tasks
completed_tasks_lock = Lock()

def run_script(total_num_tasks_param, detector, progress_bar, total_time_label, average_time_label, status_label, log_panel, num_workers):
    global cancel_operation, operation_complete, start_time, total_num_tasks  # declare global variables
    total_num_tasks = total_num_tasks_param  # assign the parameter value to the global variable
    
    cancel_operation = False  # Reset the cancel flag at the start of the operation
    
    # Initialize variables
    completed_tasks = 0
    start_time = time.time()
    
    log_message("Starting run_script function.", log_panel)
    
    # Update status label to indicate that the operation is in progress
    status_label.config(text="Operation in Progress")
    
    # Function to update all monitoring pieces
    def update_monitoring_pieces():
        global total_num_tasks  # declare global variable within nested function
        if not operation_complete:  # Check the flag before updating
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Update the progress bar and percentage
            progress_bar["value"] = (completed_tasks / total_num_tasks) * 100
            percentage = (completed_tasks / total_num_tasks) * 100 if total_num_tasks > 0 else 0
            percentage_label.config(text=f"{percentage:.2f}%")
            
            # Update the total elapsed time
            total_time_label.config(text=f"Total Elapsed Time: {elapsed_time:.2f} seconds")
            
            # Update the average time label
            if completed_tasks > 0:
                average_time = elapsed_time / completed_tasks
                average_time_label.config(text=f"Average Time for Each Spectrogram: {average_time:.2f} seconds")
            
            root.after(100, update_monitoring_pieces)
    
    # Start the function to update all monitoring pieces
    root.after(0, update_monitoring_pieces)
    
    while completed_tasks < total_num_tasks and not cancel_operation:
        remaining_tasks = total_num_tasks - completed_tasks
        current_batch_size = min(num_workers, remaining_tasks)
        
        tasks = [(random_gps_time(), detector, progress_bar) for _ in range(current_batch_size)]
        
        with ThreadPoolExecutor(max_workers=current_batch_size) as executor:
            executor.map(lambda x: download_file(*x), tasks)
            if cancel_event.is_set():
                executor.shutdown(wait=False)
                break
        
        completed_tasks += current_batch_size
        
        log_message(f"Completed {completed_tasks} of {total_num_tasks} tasks.", log_panel)
    
    # Update the global operation_complete flag to indicate the operation is complete
    if not cancel_operation:
        operation_complete = True
        status_label.config(text="Operation Completed")
    else:
        operation_complete = False  # Reset the flag if operation was cancelled
        status_label.config(text="Operation Cancelled")

    log_message("Exiting run_script function.", log_panel)
    
def increment_completed_tasks():
    global completed_tasks
    with completed_tasks_lock:
        completed_tasks += 1

def execute_script(num_tasks_entry, progress_bar, total_time_label, average_time_label, status_label, log_panel):
    log_message("Starting execute_script function.\n", log_panel)
    global selected_detector, num_workers_selected  # Use global values
    
    # Remove all files in the 'Raw Data Files' folder
    for filename in os.listdir(raw_data_folder):
        file_path = os.path.join(raw_data_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    num_tasks = int(num_tasks_entry.get())
    if num_tasks > 0:
        
        # Reset the progress bar
        progress_bar["value"] = 0
        percentage_label.config(text="0%")
        
        # Create a thread to run the script
        thread = Thread(target=run_script, args=(num_tasks, selected_detector, progress_bar, total_time_label, average_time_label, status_label, log_panel, num_workers_selected))
        thread.daemon = True  # Daemonize thread
        thread.start()
    else:
        messagebox.showwarning("Invalid Input", "Please enter a positive number of Spectrograms")
    log_message("Finished execute_script function.\n", log_panel)
    
def log_message(message, log_panel):
    log_panel.insert(tk.END, f"{message}")
    log_panel.yview(tk.END)
    
def shutdown():
    os._exit(0)


spectrogram_folder = ensure_folder_exists(directories['spectrograms'])
raw_data_folder = ensure_folder_exists(directories['raw_data'])
strain_path = ensure_folder_exists(directories['data'])

# GUI Setup
root = tk.Tk()
root.title("Gravitational Wave Data Harvester")

# Function to toggle fullscreen on/off
def toggle_fullscreen(event=None):
    root.attributes('-fullscreen', not root.attributes('-fullscreen'))

# Bind the toggle function to F11
root.bind("<F11>", toggle_fullscreen)

# Create a menu bar
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# Add "File" and "Help" menus
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open Spectrogram Folder", command=lambda: open_folder(spectrogram_folder))

help_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="Help", command=show_help_dialog)

# Parameters Frame
frame_params = ttk.LabelFrame(root, text="Parameters", padding="10")
frame_params.grid(row=0, column=0, columnspan=2, sticky='ew', padx=10, pady=5)

num_tasks_label = ttk.Label(frame_params, text="Total number of tasks:")
num_tasks_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)

num_tasks_entry = ttk.Entry(frame_params, width=10)
num_tasks_entry.grid(row=0, column=1, padx=5, pady=5)

execute_button = ttk.Button(frame_params, text="Execute Script", command=lambda: execute_script(num_tasks_entry, progress_bar, total_time_label, average_time_label, status_label, log_panel))
execute_button.grid(row=0, column=2, padx=5, pady=5)

options_button = ttk.Button(frame_params, text="Options", command=show_options_dialog)
options_button.grid(row=0, column=3, padx=5, pady=5)

help_button = ttk.Button(frame_params, text="Help", command=show_help_dialog)
help_button.grid(row=0, column=4, padx=5, pady=5)

shutdown_button = ttk.Button(frame_params, text="Shutdown", command=shutdown)
shutdown_button.grid(row=0, column=7, padx=5, pady=5)

# Monitoring Frame
frame_monitor = ttk.LabelFrame(root, text="Monitoring", padding="10")
frame_monitor.grid(row=1, column=0, columnspan=2, sticky='ew', padx=10, pady=5)  # Changed pack to grid

progress_bar = ttk.Progressbar(frame_monitor, orient=tk.HORIZONTAL, length=500, mode="determinate")
progress_bar.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky=tk.EW)  # spans across 2 columns

percentage_label = ttk.Label(frame_monitor, text="0%")
percentage_label.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)

status_label = ttk.Label(frame_monitor, text="Status: Not Started")
status_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky=tk.W)  # spans across 3 columns

total_time_label = ttk.Label(frame_monitor, text="Total Elapsed Time: 0.00 seconds")
total_time_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

average_time_label = ttk.Label(frame_monitor, text="Average Time for Each File: 0.00 seconds")
average_time_label.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

spectrogram_count_label = ttk.Label(frame_monitor, text="Number of Strain Files: 0")
spectrogram_count_label.grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)

raw_data_count_label = ttk.Label(frame_monitor, text="Number of Raw Data Files: 0")
raw_data_count_label.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)



# Logs Frame
frame_logs = ttk.LabelFrame(root, text="Logs", padding="10")
frame_logs.grid(row=2, column=1, sticky='nsew', padx=10, pady=5)  # Changed pack to grid


log_panel = Text(frame_logs, height=10, width=50)
log_panel.grid(sticky='nsew')  # Changed pack to grid

# Configure the grid to expand properly
root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(1, weight=1)
frame_logs.grid_rowconfigure(0, weight=1)
frame_logs.grid_columnconfigure(0, weight=1)

# Update Threads
update_thread = Thread(target=update_count, args=(spectrogram_count_label, raw_data_count_label, strain_path, raw_data_folder))
update_thread.daemon = True
update_thread.start()

# Start the main event loop
root.mainloop()
