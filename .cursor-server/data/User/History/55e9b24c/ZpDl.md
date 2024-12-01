# Qwen-32B Preview with Vision Capabilities

## Overview
This repository combines the power of QwQ-32B-Preview language model with CogVLM2's vision capabilities to create a powerful multimodal AI system capable of understanding and processing both text and images.

## System Requirements
- Linux operating system with NVIDIA GPU
- CUDA-compatible GPU with at least 16GB VRAM (for CogVLM2-int4)
- Python 3.8 or higher
- At least 100GB of free disk space for model storage
- CUDA toolkit 11.8 or higher

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vicmuchina/qwen-32b-preview-with-vision-capabilities.git
cd qwen-32b-preview-with-vision-capabilities
```

### 2. Set Up Python Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
First, ensure you have a clean environment:
```bash
pip uninstall -y torch torchvision xformers bitsandbytes
```

Then install the dependencies:
```bash
# Install PyTorch and torchvision with CUDA support
pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install xformers
pip install --no-cache-dir xformers==0.0.22

# Install bitsandbytes with CUDA support
pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui

# Install other dependencies
pip install -r requirements.txt
pip install -e .
```

### 4. Verify Installation
```python
import torch
import xformers
import torchvision
import bitsandbytes as bnb

print(f"PyTorch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
print(f"bitsandbytes CUDA available: {bnb.COMPILED_WITH_CUDA}")
```

### 5. Download Models
```bash
python src/multimodal_llm/download_models.py
```

This will download:
- QwQ-32B-Preview: A 32B parameter language model
- CogVLM2: Vision-language model (int4 quantized version)

## Model Details

### QwQ-32B-Preview
- 32.5B parameters
- 64 layers
- 40 attention heads for Q and 8 for KV
- Context length: 32,768 tokens
- Supports both English and Chinese

### CogVLM2
- Based on LLaMA3-8B-Instruct
- Supports 8K content length
- Image resolution up to 1344 x 1344
- Int4 quantized for efficient memory usage
- Requires only 16GB GPU memory

## Usage

### Start Interactive Chat
```bash
cd qwen-32b-preview-with-vision-capabilities
PYTHONPATH=src python -m multimodal_llm.interactive_chat --interactive
```

### Available Commands
- `/image <path>` : Load and analyze an image
- `/clear` : Clear current image
- `/exit` : Exit the chat

### Features
- Text and image processing capabilities
- High-resolution image understanding
- Long context support
- Efficient memory usage through model quantization
- Multilingual support (English and Chinese)

## Troubleshooting

### Common Issues

1. **bitsandbytes CUDA not available**
   ```bash
   pip uninstall -y bitsandbytes
   pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
   ```

2. **ModuleNotFoundError: No module named 'xformers'**
   ```bash
   pip uninstall -y torch torchvision xformers
   pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
   pip install --no-cache-dir xformers==0.0.22
   ```

3. **CUDA version mismatch**
   - Make sure you have CUDA 11.8 installed
   - Check CUDA version: `nvidia-smi` or `nvcc --version`
   - Install correct PyTorch version for your CUDA version

4. **GPU Memory Issues**
   - Free up GPU memory
   - Close other applications using GPU
   - Monitor GPU usage: `nvidia-smi`

5. **Import or Path Issues**
   - Always run from project root with `PYTHONPATH=src`
   - Check if all `__init__.py` files are present
   - Verify installation with `pip list`

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Specify your license]

## Contact
- **Author**: Victor Muchina
- **Email**: vicmuchina1234@gmail.com
- **GitHub**: [vicmuchina](https://github.com/vicmuchina)

## Acknowledgments
- Thanks to the original Qwen team for their foundational work
- [Add any other acknowledgments]

## Citation
If you use this code in your research, please cite:
```
[Add citation information]
```
