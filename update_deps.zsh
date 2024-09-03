set -ex

# Update deps
echo "Updating dependencies..."
micromamba update -a --strict-channel-priority
pip install --upgrade wandb opencv-python-headless

# Export environment
echo "\nExporting environment..."
micromamba env export >environment.yaml
pip list --format=freeze >requirements.txt

# Install project
echo "\nInstalling editable project..."
pip install -e .
