"""
hyperspeed/cli.py
Command line interface
"""

import sys
from .core import HyperSpeedEngine, convert_model, HyperSpeedFile
import json

def main():
    if len(sys.argv) < 2:
        print("""HyperSpeed - Ultra-fast CPU LLM inference

Usage:
  hyperspeed convert <input.safetensors> <output.hyper>
  hyperspeed info <model.hyper>
  
Python API:
  from hyperspeed import HyperSpeedEngine, convert_model
  
  convert_model("model.safetensors", "model.hyper")
  engine = HyperSpeedEngine("model.hyper")
  weights = engine.get_weight("transformer.layers.0.weight")
""")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "convert":
        if len(sys.argv) < 4:
            print("Usage: hyperspeed convert <input.safetensors> <output.hyper>")
            return
        convert_model(sys.argv[2], sys.argv[3])
    
    elif cmd == "info":
        if len(sys.argv) < 3:
            print("Usage: hyperspeed info <model.hyper>")
            return
        
        hsf = HyperSpeedFile()
        hsf.load(sys.argv[2])
        print("\n=== HyperSpeed Model ===")
        print(json.dumps(hsf.metadata, indent=2))
        print(f"\nTensors: {len(hsf.weights)}")
    
    else:
        print(f"Unknown command: {cmd}")

if __name__ == "__main__":
    main()
