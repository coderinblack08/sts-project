# STS Project

## Local Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:

```bash
uv sync
```

3. Create `.env` file with your OpenAI API key:

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

4. Run the project:

```bash
uv run python main.py
```

## Cloud VM Setup

For deploying on a cloud VM (AWS, GCP, Azure, etc.), use the automated setup script:

### Quick Start

1. SSH into your VM:

```bash
ssh user@your-vm-ip
```

2. Clone the repository:

```bash
git clone <your-repo-url>
cd sts-project
```

3. Run the setup script:

```bash
./setup_vm.sh
```

The script will:

- Update system packages
- Install uv
- Install all Python dependencies
- Prompt for your OpenAI API key and save it to `.env`
- Verify the installation

### Manual Cloud Setup

If you prefer manual setup or need to customize the installation:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies
uv sync

# Set up API key
read -p "Enter OpenAI API key: " -s api_key
echo "OPENAI_API_KEY=$api_key" > .env
```

## Usage

### Running Scripts

```bash
uv run python main.py
```

### Jupyter Notebooks

```bash
uv run jupyter notebook
```

## Project Structure

- `main.py` - Main entry point
- `notebooks/` - Jupyter notebooks for data generation and analysis
- `results/` - Generated datasets and activations
- `setup_vm.sh` - Automated cloud VM setup script
