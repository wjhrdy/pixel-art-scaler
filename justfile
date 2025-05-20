# Pixel Art Scaler justfile

# Default recipe to run when just is called without arguments
default:
    @just --list

# Setup development environment
setup:
    uv venv
    @echo "To activate the virtual environment, run:"
    @echo "source .venv/bin/activate  # On Unix/macOS"
    @echo ".venv\\Scripts\\activate  # On Windows"

# Install dependencies directly
install:
    uv pip install pillow numpy pyside6 scipy

# Run the CLI application
run-cli IMAGE_PATH:
    python cli.py {{IMAGE_PATH}}

# Run the PySide6 GUI application
run-gui:
    python gui.py

# Clean up environment
clean:
    rm -rf .venv
    rm -rf *.egg-info
    rm -rf __pycache__
    rm -rf ./**/__pycache__

# Full setup and run CLI (setup, install)
quickstart:
    just setup
    source .venv/bin/activate && just install

# Process an image with the CLI tool
process IMAGE_PATH:
    source .venv/bin/activate && python cli.py {{IMAGE_PATH}}