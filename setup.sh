#!/bin/bash

set -e  

# Check if requirements.txt exists before installing dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    echo "Installation complete."
else
    echo "Error: requirements.txt not found!"
    exit 1  # Exit with error status
fi
