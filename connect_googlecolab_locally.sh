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

if [ -z "$1" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
    exit 0
fi

ENV_NAME=$1

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' found."
else
    echo "Environment '$ENV_NAME' not found. Creating it..."
    conda create -n "$ENV_NAME" python=3.10 pip -y || exit 1
fi

echo "Activating '$ENV_NAME'..."
conda activate "$ENV_NAME" || { echo "Could not activate '$ENV_NAME'."; exit 1; }

PY="$CONDA_PREFIX/bin/python"
if [ ! -x "$PY" ]; then
    echo "'$ENV_NAME' has no Python of its own. Installing python=3.10 + pip into it..."
    conda install -n "$ENV_NAME" python=3.10 pip -y || exit 1
fi
echo "Using Python: $PY ($("$PY" --version))"

pip_env() { "$PY" -m pip "$@"; }

if ! pip_env show notebook 2>/dev/null | grep -q "^Version: 6" || ! pip_env show jupyter_http_over_ws > /dev/null 2>&1; then
    echo "Fixing dependencies (downgrading Notebook, installing extension)..."

    pip_env uninstall -y jupyterlab notebook jupyter-server

    pip_env install "notebook==6.5.7" "jupyter-server<2.0.0" jupyter_http_over_ws || exit 1

    "$CONDA_PREFIX/bin/jupyter" serverextension enable --py jupyter_http_over_ws --sys-prefix
else
    echo "Required packages (Notebook 6 & Colab Extension) are present."
fi

echo "Launching Jupyter for Google Colab..."
echo "COPY THE URL BELOW AND PASTE IT INTO COLAB:"
echo "---------------------------------------------------"
exec "$CONDA_PREFIX/bin/jupyter" notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --NotebookApp.port_retries=0 \
    --no-browser