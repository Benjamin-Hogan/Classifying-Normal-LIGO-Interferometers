#!/bin/bash

# Ensure Homebrew is up to date
brew update

# Install Python 3.11 with Homebrew
echo "Installing Python 3.11..."
brew install python@3.11

# Link Python 3.11 (if necessary) and verify the installation
brew link python@3.11 --force --overwrite
python3.11 --version

# Upgrade pip to its latest version using the newly installed Python version
echo "Upgrading pip..."
python3.11 -m pip install --upgrade pip

# Install required Python packages using the upgraded pip
echo "Installing Python packages..."
python3.11 -m pip install numpy scipy matplotlib requests gwpy tqdm sounddevice tensorflow pandas opencv-python-headless Pillow pygame openpyxl reportlab

#Installing Tkinter Python Package
brew install python-tk
# Verify tkinter installation
echo "Verifying tkinter installation..."
python3.11 -m tkinter

echo "Installation complete."