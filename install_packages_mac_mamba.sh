#!/bin/bash

# Ensure conda is up to date
echo "Updating conda..."
conda update -n base -c defaults conda -y

# Install mamba in the base environment if it isn't already installed
echo "Installing mamba in the base environment..."
conda install mamba -n base -c conda-forge -y

# Use mamba to install or update Python to 3.11 in the base environment
echo "Ensuring Python 3.11 is installed..."
mamba install python=3.11 -n base -c conda-forge -y

# Use mamba to install the required packages in the base environment
echo "Installing Python packages with mamba..."
mamba install numpy scipy matplotlib requests gwpy tqdm sounddevice tensorflow pandas opencv-python-headless pillow pygame openpyxl reportlab tk -c conda-forge -y

# Verify the installation of Python and packages
echo "Verifying Python installation..."
python --version
echo "Python packages installed."

echo "Installation complete."
