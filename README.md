# Project Name: Classifying Normal for LIGO Interferometers

## Description
This project analyzes and classifies the normal range for LIGO's gravitational wave detectors. 
This is done using neural networks and more specifically an autoencoder. 
It includes scripts for harvesting data straight from lIGO, normalizing the data, training the neural network, and testing it. 

## Prerequisites
Before you begin, ensure you have met the following requirements:
- macOS Operating System

## Installation
Follow these steps to install the necessary environment and dependencies for the project:

1. **Clone git Repository:**
   - Open a terminal and navigate to the directory to put the repository on your local computer. It will look something like user@user_device/documents
   - clone the repository using the following command:
     ```bash
     git clone https://github.com/Benjamin-Hogan/Classifying-Normal-LIGO-Interferometers.git
     ```
2. **Python and Dependencies Installation:**
   - Open a terminal and navigate to the project directory. It will look something like user@user_device/documents/Classifying-Normal-LIGO-interferometers
   - The hierarchy of where the terminal will be:
     ```bash
      ├── LICENSE
      ├── README.md
      ├── code
      │   ├── CNN-Tester.py
      │   ├── GUI-Harvester.py
      │   ├── Neural-Network.py
      │   └── Normalization.py
      ├── install_packages_mac_mamba.sh
      └── install_packages_mac_standard.sh
   - Run the script `install_packages_mac_standard.sh` or if you prefer using mamba use the same steps but use `install_packages_mamba.sh`. This will install Python 3.11 and all required Python packages:
     ```bash
     ./install_packages_mac.sh
     ```   
This script will install `numpy`, `scipy`, `matplotlib`, `requests`, `gwpy`, `tqdm`, `sounddevice`, `tensorflow`, `pandas`, `opencv-python-headless`, `Pillow`, `pygame`, `openpyxl`, `reportlab`, and the `tkinter` Python package using Homebrew and pip.

2. **Ensure All Scripts are Executable:**
   - For each `.py` script file (`CNN-Tester.py`, `GUI-Harvester.py`, `Neural-Network.py`, `Normalization.py`), change the permission to make sure they are executable:
     ```bash
     chmod +x *.py
     ```

## Usage
To run the project successfully, follow these steps:

1. **Graphical User Interface for Data Harvesting:**
   - Execute `GUI-Harvester.py` to start the graphical user interface for data harvesting:
     ```bash
     ./GUI-Harvester.py
     ```
     Alternatively: Use whatever IDE you prefer and run it there (I use VScode)
   - Follow the GUI prompts to download and process the gravitational wave data.

2. **Data Normalization:**
   - Run `Normalization.py` to normalize your data. This script prepares your data for neural network training by normalizing it.
     ```bash
     ./Normalization.py
     ```

3. **Neural Network Training:**
   - Once the data is harvested and normalized, you can train the neural network by running `Neural-Network.py`:
     ```bash
     ./Neural-Network.py
     ```
   - This script trains a neural network model on the prepared data. Monitor the training process and evaluate the model's performance.

4. **CNN Testing (Optional):**
   - If you have a convolutional neural network (CNN) model to test after the neural network training script has been run, run `CNN-Tester.py`:
     ```bash
     ./CNN-Tester.py
     ```
   - This optional step is for testing and evaluating CNN models.

## Support
For support, email benhogan01@yahoo.com or open an issue on the GitHub repository.

## Authors and Acknowledgment
Thank you to the contributors of this project.
- Dr. Andri M. Gretarsson (Research Professor ERAU) 
- Benjamin E. Hogan (Undergraduate Researcher ERAU)

## Project Status
The project is in active development. Future updates will focus on improving the neural network model, expanding the data processing, and testing capabilities.
