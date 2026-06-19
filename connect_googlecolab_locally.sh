#!/bin/bash
# Usage: ./connect_googlecolab_locally.sh <env_name>

# Function to display usage instructions
show_help() {
    echo "Usage: $0 <env_name>"
    echo ""
    echo "Description:"
    echo "  This script prepares a Conda environment and launches a local"
    echo "  Jupyter runtime compatible with Google Colab."
    echo ""
    echo "Arguments:"
    echo "  <env_name>      The name of the Conda environment to use/create."
    echo ""
    echo "Options:"
    echo "  -h, --help      Show this help message."
    echo ""
    echo "Example:"
    echo "  $0 my_thesis_env"
    echo ""
}

# Check if no arguments provided or if help flag is used
if [ -z "$1" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
    exit 0
fi

ENV_NAME=$1

# 1. Initialize Conda for bash script usage
# This allows 'conda activate' to work inside this script
eval "$(conda shell.bash hook)"

# 2. Check if environment exists
if conda env list | grep -q "\b$ENV_NAME\b"; then
    echo "✅ Conda environment '$ENV_NAME' found."
else
    echo "⚠️ Environment '$ENV_NAME' not found. Creating it..."
    conda create -n $ENV_NAME python=3.10 -y
fi

# 3. Activate the environment
echo "🔄 Activating '$ENV_NAME'..."
conda activate $ENV_NAME

# 4. Check/Install Dependencies
# We check if notebook is installed AND if it is the correct version (starts with 6)
if ! pip show notebook | grep -q "Version: 6" || ! pip show jupyter_http_over_ws > /dev/null; then
    echo "🛠 Fixing dependencies (downgrading Notebook, installing extension)..."
    
    # Uninstall conflicting packages first to be safe
    pip uninstall -y jupyterlab notebook jupyter-server
    
    # Install specific compatible versions
    # We explicitly install jupyter-server<2.0.0 to prevent the AttributeError you saw earlier
    pip install "notebook==6.5.7" "jupyter-server<2.0.0" jupyter_http_over_ws
    
    # Enable the extension
    jupyter serverextension enable --py jupyter_http_over_ws
else
    echo "✅ Required packages (Notebook 6 & Colab Extension) are present."
fi

# 5. Launch
echo "🚀 Launching Jupyter for Google Colab..."
echo "COPY THE URL BELOW AND PASTE IT INTO COLAB:"
echo "---------------------------------------------------"
jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com' --port=8888 --no-browser