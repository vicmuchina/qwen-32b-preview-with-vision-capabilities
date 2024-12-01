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

### 3. Install PyTorch with CUDA support
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 5. Download Models
```bash
python src/multimodal_llm/download_models.py
```

This will download:
- QwQ-32B-Preview: A 32B parameter language model
- CogVLM2: Vision-language model (int4 quantized version)

## Dependencies
The main dependencies include:
- PyTorch 2.2.1
- Transformers 4.37.2+
- xformers 0.0.23+
- accelerate 0.27.1+
- bitsandbytes 0.42.0+
- Other dependencies listed in requirements.txt

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
python src/multimodal_llm/interactive_chat.py --interactive
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
If you encounter any issues:
1. Make sure you have CUDA toolkit installed and compatible with PyTorch
2. Ensure all dependencies are installed correctly
3. Check GPU memory availability
4. Verify the image path when using the `/image` command

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
