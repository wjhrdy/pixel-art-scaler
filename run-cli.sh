#!/bin/bash

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Please install it first from https://docs.astral.sh/uv/"
    exit 1
fi

# Install dependencies using uv
uv pip install -e .

# Run the CLI version
uvx pixel-art-scaler-cli "$@"
