#!/usr/bin/env python3
"""Independent NumPy reference forward pass for Qwen3 F32 GGUF checkpoints.

Written from the Hugging Face Qwen3 definition (rotate_half RoPE, per-head q/k RMSNorm, GQA,
SwiGLU), in float64, with a self-contained GGUF reader. It shares no code with the F# runtime and
is used to produce the golden logits consumed by tests/NumericalParityTests.

usage: qwen3_reference.py MODEL.gguf OUTPUT.json TOKEN [TOKEN ...]
"""
import json
import struct
import sys

import numpy as np

GGUF_TYPES = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def read_gguf(path):
    with open(path, "rb") as f:
        data = f.read(64 << 20)  # header + metadata fit comfortably in 64 MiB
    pos = 0

    def take(fmt):
        nonlocal pos
        size = struct.calcsize(fmt)
        (value,) = struct.unpack_from(fmt, data, pos)
        pos += size
        return value

    def string():
        nonlocal pos
        n = take("<Q")
        s = data[pos:pos + n].decode("utf-8", errors="replace")
        pos += n
        return s

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            et = take("<I")
            n = take("<Q")
            return [value(et) for _ in range(n)]
        return take(GGUF_TYPES[t])

    assert data[:4] == b"GGUF", "not a GGUF file"
    pos = 4
    take("<I")  # version
    n_tensors = take("<Q")
    n_kv = take("<Q")
    meta = {}
    for _ in range(n_kv):
        key = string()
        meta[key] = value(take("<I"))
    tensors = {}
    for _ in range(n_tensors):
        name = string()
        dims = [take("<Q") for _ in range(take("<I"))]
        dtype = take("<I")
        offset = take("<Q")
        tensors[name] = (dims, dtype, offset)
    align = meta.get("general.alignment", 32)
    base = (pos + align - 1) // align * align
    mm = np.memmap(path, dtype=np.uint8, mode="r")

    def tensor(name):
        dims, dtype, offset = tensors[name]
        assert dtype == 0, f"{name}: only F32 checkpoints are supported by the reference"
        count = int(np.prod(dims))
        flat = mm[base + offset: base + offset + 4 * count].view(np.float32)
        # ggml dims are fastest-first: a weight W(out, in) has dims [in, out].
        return np.asarray(flat.reshape(list(reversed(dims))), dtype=np.float64)

    return meta, tensor


def rms_norm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def main():
    model_path, out_path = sys.argv[1], sys.argv[2]
    tokens = [int(t) for t in sys.argv[3:]]
    meta, tensor = read_gguf(model_path)
    arch = meta["general.architecture"]
    assert arch == "qwen3", arch
    n_layers = meta[f"{arch}.block_count"]
    n_heads = meta[f"{arch}.attention.head_count"]
    n_kv = meta[f"{arch}.attention.head_count_kv"]
    head_dim = meta[f"{arch}.attention.key_length"]
    eps = meta[f"{arch}.attention.layer_norm_rms_epsilon"]
    theta = meta[f"{arch}.rope.freq_base"]

    T = len(tokens)
    embedding = tensor("token_embd.weight")
    x = embedding[tokens]  # (T, dim)
    positions = np.arange(T, dtype=np.float64)
    inv_freq = 1.0 / theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    angles = np.outer(positions, inv_freq)  # (T, hd/2)
    cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)[:, None, :]
    sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)[:, None, :]
    causal = np.triu(np.full((T, T), -np.inf), k=1)

    for layer in range(n_layers):
        p = f"blk.{layer}."
        h = rms_norm(x, tensor(p + "attn_norm.weight"), eps)
        q = (h @ tensor(p + "attn_q.weight").T).reshape(T, n_heads, head_dim)
        k = (h @ tensor(p + "attn_k.weight").T).reshape(T, n_kv, head_dim)
        v = (h @ tensor(p + "attn_v.weight").T).reshape(T, n_kv, head_dim)
        q = rms_norm(q, tensor(p + "attn_q_norm.weight"), eps)
        k = rms_norm(k, tensor(p + "attn_k_norm.weight"), eps)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        group = n_heads // n_kv
        out = np.empty((T, n_heads, head_dim))
        for hh in range(n_heads):
            kv = hh // group
            scores = q[:, hh, :] @ k[:, kv, :].T / np.sqrt(head_dim) + causal
            scores -= scores.max(axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs /= probs.sum(axis=-1, keepdims=True)
            out[:, hh, :] = probs @ v[:, kv, :]
        x = x + out.reshape(T, n_heads * head_dim) @ tensor(p + "attn_output.weight").T
        h = rms_norm(x, tensor(p + "ffn_norm.weight"), eps)
        gate = h @ tensor(p + "ffn_gate.weight").T
        up = h @ tensor(p + "ffn_up.weight").T
        x = x + ((gate / (1.0 + np.exp(-gate))) * up) @ tensor(p + "ffn_down.weight").T

    x = rms_norm(x, tensor("output_norm.weight"), eps)
    try:
        classifier = tensor("output.weight")
    except KeyError:
        classifier = embedding
    logits = x @ classifier.T  # (T, vocab)

    positions_out = []
    for t in range(T):
        top = np.argsort(-logits[t])[:10]
        positions_out.append({"top_ids": [int(i) for i in top], "top_logits": [float(logits[t, i]) for i in top]})
    with open(out_path, "w") as f:
        json.dump({"model": model_path.split("/")[-1], "tokens": tokens, "positions": positions_out}, f, indent=1)
    print(f"wrote {out_path}: {T} positions, argmax {[p['top_ids'][0] for p in positions_out]}")


if __name__ == "__main__":
    main()
