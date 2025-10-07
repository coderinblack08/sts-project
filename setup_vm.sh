#!/bin/bash

set -e

echo "======================================"
echo "  STS Project VM Setup Script"
echo "======================================"
echo ""

echo "[1/5] Updating system packages..."
sudo apt-get update -qq

echo ""
echo "[2/5] Installing uv..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    echo "✓ uv installed successfully"
else
    echo "✓ uv is already installed"
fi

echo ""
echo "[3/5] Installing Python and project dependencies..."
cd "$(dirname "$0")"
uv sync

echo ""
echo "[4/5] Setting up environment variables..."
if [ -f ".env" ]; then
    echo "⚠ .env file already exists"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping .env setup"
        ENV_SETUP=false
    else
        ENV_SETUP=true
    fi
else
    ENV_SETUP=true
fi

if [ "$ENV_SETUP" = true ]; then
    read -p "Enter your OpenAI API key: " -s OPENAI_API_KEY
    echo
    
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "⚠ No API key provided. You can add it manually to .env later"
    else
        echo "OPENAI_API_KEY=$OPENAI_API_KEY" > .env
        echo "✓ API key saved to .env"
    fi
fi

echo ""
echo "[5/5] Verifying installation..."
if uv run python -c "import torch; import transformer_lens; import openai; print('All imports successful')" 2>/dev/null; then
    echo "✓ All dependencies installed correctly"
else
    echo "⚠ Warning: Some dependencies may not have installed correctly"
fi

echo ""
echo "======================================"
echo "  Setup Complete!"
echo "======================================"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Or run commands directly with:"
echo "  uv run python main.py"
echo ""

