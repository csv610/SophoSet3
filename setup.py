"""Setup configuration for SophoSet3."""

from setuptools import setup, find_packages

setup(
    name="sophoset",
    version="0.1.0",
    description="Unified Hugging Face datasets management framework for evaluating vision-language models (VLMs) and large language models (LLMs)",
    author="SophoSet Contributors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "datasets>=2.0.0",
        "pillow>=9.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "transformers>=4.0.0",
        "python-dotenv>=0.19.0",
        "tqdm>=4.62.0",
        "requests>=2.26.0",
        "lmdb>=1.0.0",
        "huggingface-hub>=0.11.0",
        "pydantic>=1.8.0",
        "litellm>=0.1.0",
    ],
    extras_require={
        "ui": [
            "streamlit>=1.0.0",
            "gradio>=3.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "isort>=5.10.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
