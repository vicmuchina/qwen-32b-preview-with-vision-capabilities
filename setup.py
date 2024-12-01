from setuptools import setup, find_packages

setup(
    name="multimodal_llm",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.34.0',
        'accelerate',
        'sentencepiece',
        'pillow',
        'llama-cpp-python',
        'opencv-python',
        'timm',
        'einops',
        'requests',
        'tqdm',
        'bitsandbytes>=0.41.0',
        'transformers_stream_generator',
        'regex'
    ]
)
