"""
hyperspeed/core.py
Core engine with inference
"""

import struct
import numpy as np
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterator

MAGIC = b"HYPR"
VERSION = 1
QUANT_BLOCK_SIZE = 32

class AdaptiveQuantizer:
    @staticmethod
    def quantize(weights: np.ndarray) -> Tuple[bytes, Dict]:
        """Adaptive quantization: 2/4/8-bit based on importance"""
        importance = np.abs(weights) * (1 + np.var(weights))
        flat = weights.flatten()
        flat_imp = importance.flatten()
        
        sorted_imp = np.sort(flat_imp)
        t_crit = sorted_imp[int(0.95 * len(sorted_imp))]
        t_imp = sorted_imp[int(0.80 * len(sorted_imp))]
        
        bits = np.ones(len(flat), dtype=np.uint8) * 2
        bits[flat_imp >= t_imp] = 4
        bits[flat_imp >= t_crit] = 8
        
        quantized = bytearray()
        scales, zeros = [], []
        
        for i in range(0, len(flat), QUANT_BLOCK_SIZE):
            block = flat[i:i+QUANT_BLOCK_SIZE]
            b_bits = int(np.median(bits[i:i+QUANT_BLOCK_SIZE]))
            
            mn, mx = block.min(), block.max()
            scale = (mx - mn) / (2**b_bits - 1) if mx != mn else 1.0
            
            quant = np.clip(np.round((block - mn) / scale), 0, 2**b_bits - 1).astype(np.uint8)
            scales.append(scale)
            zeros.append(mn)
            quantized.extend(quant.tobytes())
        
        return bytes(quantized), {
            'shape': weights.shape,
            'scales': scales,
            'zeros': zeros,
            'block_size': QUANT_BLOCK_SIZE
        }
    
    @staticmethod
    def dequantize(data: bytes, meta: Dict) -> np.ndarray:
        """Dequantize adaptive format"""
        quant = np.frombuffer(data, dtype=np.uint8)
        result = np.zeros(np.prod(meta['shape']), dtype=np.float32)
        
        for i, (scale, zero) in enumerate(zip(meta['scales'], meta['zeros'])):
            start = i * meta['block_size']
            end = start + meta['block_size']
            result[start:end] = quant[start:end].astype(np.float32) * scale + zero
        
        return result.reshape(meta['shape'])


class HyperSpeedFile:
    def __init__(self):
        self.metadata = {}
        self.weights = {}
        self.quant_meta = {}
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            f.write(MAGIC)
            f.write(struct.pack('I', VERSION))
            
            meta_json = json.dumps(self.metadata).encode()
            f.write(struct.pack('I', len(meta_json)))
            f.write(meta_json)
            
            quant_json = json.dumps(self.quant_meta).encode()
            f.write(struct.pack('I', len(quant_json)))
            f.write(quant_json)
            
            f.write(struct.pack('I', len(self.weights)))
            for name, data in self.weights.items():
                name_bytes = name.encode()
                f.write(struct.pack('I', len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack('I', len(data)))
                f.write(data)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            if f.read(4) != MAGIC:
                raise ValueError("Invalid HyperSpeed file")
            if struct.unpack('I', f.read(4))[0] != VERSION:
                raise ValueError("Unsupported version")
            
            meta_len = struct.unpack('I', f.read(4))[0]
            self.metadata = json.loads(f.read(meta_len))
            
            quant_len = struct.unpack('I', f.read(4))[0]
            self.quant_meta = json.loads(f.read(quant_len))
            
            weight_count = struct.unpack('I', f.read(4))[0]
            for _ in range(weight_count):
                name_len = struct.unpack('I', f.read(4))[0]
                name = f.read(name_len).decode()
                data_len = struct.unpack('I', f.read(4))[0]
                self.weights[name] = f.read(data_len)


def convert_model(input_path: str, output_path: str, verbose: bool = True):
    """Convert model to HyperSpeed format"""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("pip install safetensors")
    
    if verbose:
        print(f"Converting {input_path}...")
    
    start = time.time()
    hsf = HyperSpeedFile()
    hsf.metadata = {'source': input_path, 'format': 'hyperspeed', 'version': VERSION}
    
    with safe_open(input_path, framework="numpy") as f:
        keys = list(f.keys())
        for idx, key in enumerate(keys):
            if verbose:
                print(f"  [{idx+1}/{len(keys)}] {key}", end='\r')
            tensor = f.get_tensor(key)
            data, meta = AdaptiveQuantizer.quantize(tensor)
            hsf.weights[key] = data
            hsf.quant_meta[key] = meta
    
    hsf.save(output_path)
    
    if verbose:
        elapsed = time.time() - start
        orig_size = os.path.getsize(input_path) / (1024**2)
        comp_size = os.path.getsize(output_path) / (1024**2)
        print(f"\n✓ Done in {elapsed:.1f}s | {orig_size:.1f}MB → {comp_size:.1f}MB ({orig_size/comp_size:.1f}x)")


class HyperSpeedEngine:
    """Fast inference engine with streaming support"""
    
    def __init__(self, model_path: str, verbose: bool = True):
        self.model_path = model_path
        self.weights = {}
        self.tokenizer = None
        
        if verbose:
            print(f"Loading {model_path}...")
        
        start = time.time()
        hsf = HyperSpeedFile()
        hsf.load(model_path)
        
        self._quant_weights = hsf.weights
        self._quant_meta = hsf.quant_meta
        self.metadata = hsf.metadata
        
        if verbose:
            print(f"✓ Loaded in {time.time()-start:.2f}s ({len(hsf.weights)} tensors)")
    
    def get_weight(self, name: str) -> np.ndarray:
        """Get weight tensor (dequantize on demand)"""
        if name not in self.weights:
            self.weights[name] = AdaptiveQuantizer.dequantize(
                self._quant_weights[name],
                self._quant_meta[name]
            )
        return self.weights[name]
    
    def list_weights(self) -> list:
        """List all weight tensor names"""
        return list(self._quant_weights.keys())
    
    def load_tokenizer(self, model_name: str):
        """Load tokenizer from HuggingFace"""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7, stream: bool = False):
        """
        Generate text (uses transformers as backend for now)
        
        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: If True, yields tokens one by one (Iterator[str])
        
        Returns:
            str or Iterator[str]: Generated text or token stream
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            return "Error: pip install transformers"
        
        # Load model if needed
        if not hasattr(self, '_hf_model'):
            model_name = self.metadata.get('source', 'Qwen/Qwen2.5-0.5B')
            print(f"Loading HF model for inference: {model_name}")
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="cpu"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if stream:
            return self._generate_stream(inputs, max_tokens, temperature)
        else:
            outputs = self._hf_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _generate_stream(self, inputs, max_tokens: int, temperature: float) -> Iterator[str]:
        """Stream tokens one by one"""
        import torch
        
        input_ids = inputs['input_ids']
        past_key_values = None
        
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self._hf_model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_text
            
            input_ids = next_token
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
