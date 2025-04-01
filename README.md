# vLLM Installer

An enhanced installation tool for setting up [vLLM](https://github.com/vllm-project/vllm) on bare metal servers with improved user experience, verbosity, and model selection capabilities.

## Features

- **User-Friendly Installation**: Interactive prompts guide you through the setup process
- **Enhanced Model Selection**: 
  - Browse and select from a curated list of popular LLM models
  - Browse trending models from Hugging Face
  - Add custom models with proper configuration
- **Real-Time Feedback**: Color-coded output and progress indicators
- **System Integration**: Automatic systemd service setup for running as a service
- **Management Utilities**: Tools for managing models, services, and monitoring

## Installation

Clone this repository:

```bash
git clone https://github.com/yourusername/vllm-installer.git
cd vllm-installer
```

Make the installer executable:

```bash
chmod +x vllm_installer.sh
```

Run the installer:

```bash
./vllm_installer.sh
```

## Requirements

- Linux operating system (Ubuntu/Debian recommended)
- Python 3.8+
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed

## Model Selection

The installer provides multiple ways to select models:

1. **Basic Selection**: Choose from predefined popular models (Llama 3, Mistral, Phi-3, etc.)
2. **Advanced Selection**: Browse and select from categorized popular models
3. **Trending Models**: Browse trending models from Hugging Face

For gated models like Llama 3, you'll need a Hugging Face token.

## Installing Models Separately

You can install models separately using the included model installation script:

```bash
./scripts/vllm/install_models.sh
```

This will guide you through selecting and downloading models without reinstalling vLLM.

## Components

- `vllm_installer.sh`: Main installation script
- `scripts/vllm/`: Directory containing supporting scripts:
  - `popular_models.py`: Interactive model selection tool
  - `download_models.py`: Model downloading utility
  - `monitor_gpu.py`: GPU monitoring tool
  - `vllm_server.py`: vLLM API server script
  - `vllm_manage.sh`: Service management script
  - `install_models.sh`: Standalone model installation script

## Configuration

The installer creates a configuration directory at `/opt/vllm/config` (default) with:

- `model_config.json`: Configuration for models to be served by vLLM

## Management

After installation, you can manage your vLLM setup with:

```bash
vllm-manage [command]
```

Available commands:
- `start`: Start the vLLM service
- `stop`: Stop the vLLM service
- `restart`: Restart the vLLM service
- `status`: Check service status
- `list-models`: List configured models
- `download-models`: Download models from the configuration
- `monitor`: Monitor GPU usage in real-time
- And more...

## License

MIT

## Acknowledgments

Built on top of [vLLM](https://github.com/vllm-project/vllm), an open-source library for fast LLM inference.
