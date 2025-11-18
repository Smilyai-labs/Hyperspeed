from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hyperspeed",
    version="1.0.0",
    author="Smilyai-labs",
    description="Ultra-fast CPU LLM inference library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hyperspeed",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "convert": ["safetensors>=0.3.0"],
    },
    entry_points={
        "console_scripts": [
            "hyperspeed=hyperspeed.cli:main",
        ],
    },
)
