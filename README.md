# HyperSpeed ⚡

Ultra-fast CPU LLM inference format with adaptive quantization and cache-optimized layouts.

## Install

```bash
pip install git+https://github.com/Smilyai-labs/Hyperspeed.git
```

## Quick Start

```python
from hyperspeed import HyperSpeedEngine, convert_model

# Convert model
convert_model("model.safetensors", "model.hyper")

# Load and use
engine = HyperSpeedEngine("model.hyper")
weights = engine.get_weight("transformer.layer.0.weight")
```

## CLI

```bash
# Convert
hyperspeed convert model.safetensors model.hyper

# Info
hyperspeed info model.hyper
```

## Features

- **Adaptive Quantization**: 2/4/8-bit based on weight importance
- **Cache-Optimized Layout**: Better CPU cache utilization  
- **Fast Loading**: Lazy dequantization on-demand
- **2-4x Compression**: Smaller than original models

## Example

```python
# Download Qwen model (safetensors format)
# Convert to HyperSpeed
convert_model("qwen.safetensors", "qwen.hyper")

# Use in your inference code
engine = HyperSpeedEngine("qwen.hyper")
for name in engine.list_weights():
    w = engine.get_weight(name)
    print(f"{name}: {w.shape}")
```

## Requirements

- Python 3.8+
- numpy
- safetensors (for conversion)

## License

Apache License 2.0
