# How llm-d Distributed Inference Splits Models

## Overview

This document explains how llm-d (and vLLM) use **tensor parallelism** to split large language models across multiple GPUs for distributed inference.

---

## The Core Concept: Tensor Parallelism

The model's weight matrices are **sliced by columns/rows** and distributed across GPUs. Each GPU holds a **slice of every layer**, not separate layers.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     TENSOR PARALLELISM - COLUMN/ROW SLICING                         │
│                                                                                     │
│  Original Weight Matrix (e.g., Q projection in attention)                           │
│  Shape: [hidden_size × hidden_size] = [4096 × 4096]                                 │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                             │    │
│  │     ┌───────────┬───────────┬───────────┬───────────┐                       │    │
│  │     │           │           │           │           │                       │    │
│  │     │  Col 0-   │ Col 1024- │ Col 2048- │ Col 3072- │                       │    │
│  │     │   1023    │   2047    │   3071    │   4095    │                       │    │
│  │     │           │           │           │           │                       │    │
│  │     │  [4096 ×  │  [4096 ×  │  [4096 ×  │  [4096 ×  │                       │    │
│  │     │   1024]   │   1024]   │   1024]   │   1024]   │                       │    │
│  │     │           │           │           │           │                       │    │
│  │     └─────┬─────┴─────┬─────┴─────┬─────┴─────┬─────┘                       │    │
│  │           │           │           │           │                             │    │
│  │           ▼           ▼           ▼           ▼                             │    │
│  │        GPU 0       GPU 1       GPU 2       GPU 3                            │    │
│  │                                                                             │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
│  Each GPU has 1/4 of the weights but processes ALL tokens                           │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## What Gets Split vs What Doesn't

### Split Across GPUs (Large Matrices)

| Component | How It's Split |
|-----------|----------------|
| Attention Q, K, V projections | Split by attention heads |
| Attention Output projection | Split by rows |
| MLP Gate projection | Split by columns |
| MLP Up projection | Split by columns |
| MLP Down projection | Split by rows |
| Token embeddings | Split by columns |
| LM head (output layer) | Split by columns |

---

## Concrete Example: Llama 70B on 4 GPUs

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Layers | 80 |
| Hidden size | 8192 |
| Attention heads | 64 |
| Intermediate size (MLP) | 28672 |
| Total parameters | ~70B |
| Memory (FP16) | ~140GB |

### Distribution with tensor_parallel_size=4

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     LLAMA 70B - 4 GPU TENSOR PARALLELISM                            │
│                                                                                     │
│  With tensor_parallel_size=4:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                             │    │
│  │   GPU 0                GPU 1                GPU 2                GPU 3      │    │
│  │   ┌──────────┐        ┌──────────┐        ┌──────────┐        ┌──────────┐  │    │
│  │   │          │        │          │        │          │        │          │  │    │
│  │   │ 80 layers│        │ 80 layers│        │ 80 layers│        │ 80 layers│  │    │
│  │   │ (all!)   │        │ (all!)   │        │ (all!)   │        │ (all!)   │  │    │
│  │   │          │        │          │        │          │        │          │  │    │
│  │   │ But only │        │ But only │        │ But only │        │ But only │  │    │
│  │   │ 1/4 of   │        │ 1/4 of   │        │ 1/4 of   │        │ 1/4 of   │  │    │
│  │   │ weights  │        │ weights  │        │ weights  │        │ weights  │  │    │
│  │   │ per layer│        │ per layer│        │ per layer│        │ per layer│  │    │
│  │   │          │        │          │        │          │        │          │  │    │
│  │   │ Heads    │        │ Heads    │        │ Heads    │        │ Heads    │  │    │
│  │   │ 0-15     │        │ 16-31    │        │ 32-47    │        │ 48-63    │  │    │
│  │   │          │        │          │        │          │        │          │  │    │
│  │   │ ~35GB    │        │ ~35GB    │        │ ~35GB    │        │ ~35GB    │  │    │
│  │   └──────────┘        └──────────┘        └──────────┘        └──────────┘  │    │
│  │                                                                             │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
│  Each GPU: 140GB ÷ 4 = ~35GB weights + KV cache + activations                       │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Memory Breakdown Per GPU

| Component | Size |
|-----------|------|
| Model weights | ~35GB (140GB ÷ 4) |
| KV cache | ~8GB (varies by context length) |
| Activations | ~2GB |
| **Total** | **~45GB** |

---

## How a Forward Pass Works

### Step 1: Embedding Lookup (Parallel)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Input: "What is AI?" → Tokens: [1, 1724, 338, 319, 29902]                          │
│                                                                                     │
│    GPU 0              GPU 1              GPU 2              GPU 3                   │
│    Embed cols         Embed cols         Embed cols         Embed cols              │
│    [0:2048]           [2048:4096]        [4096:6144]        [6144:8192]             │
│        │                  │                  │                  │                   │
│        └──────────────────┴──────────────────┴──────────────────┘                   │
│                                    │                                                │
│                              ALL-GATHER                                             │
│                     (combine partial embeddings)                                    │
│                                    │                                                │
│                                    ▼                                                │
│                    Full embedding [5 tokens × 8192]                                 │
│                    (replicated on all GPUs)                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Step 2: Transformer Layers (Repeated 80 Times)

#### 2a. Attention (Parallel by Heads)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Input X: [5 × 8192] (same on all GPUs)                                             │
│                                                                                     │
│    GPU 0              GPU 1              GPU 2              GPU 3                   │
│    Q,K,V for          Q,K,V for          Q,K,V for          Q,K,V for               │
│    heads 0-15         heads 16-31        heads 32-47        heads 48-63             │
│        │                  │                  │                  │                   │
│        │    Compute attention scores and weighted values                            │
│        │    (each GPU only for its heads)                                           │
│        │                  │                  │                  │                   │
│        ▼                  ▼                  ▼                  ▼                   │
│    Partial out        Partial out        Partial out        Partial out             │
│    [5 × 2048]         [5 × 2048]         [5 × 2048]         [5 × 2048]              │
│        │                  │                  │                  │                   │
│        └──────────────────┴──────────────────┴──────────────────┘                   │
│                                    │                                                │
│                              ALL-REDUCE                                             │
│                    (sum partial outputs → full output)                              │
│                                    │                                                │
│                                    ▼                                                │
│                    Attention output [5 × 8192]                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

#### 2b. MLP (Parallel by Columns then Rows)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Input: [5 × 8192]                                                                  │
│                                                                                     │
│    GPU 0              GPU 1              GPU 2              GPU 3                   │
│    Gate+Up cols       Gate+Up cols       Gate+Up cols       Gate+Up cols            │
│    [0:7168]           [7168:14336]       [14336:21504]      [21504:28672]           │
│        │                  │                  │                  │                   │
│        ▼                  ▼                  ▼                  ▼                   │
│    Partial MLP        Partial MLP        Partial MLP        Partial MLP             │
│    [5 × 7168]         [5 × 7168]         [5 × 7168]         [5 × 7168]              │
│        │                  │                  │                  │                   │
│        │    Apply activation (SiLU), then Down projection                           │
│        │                  │                  │                  │                   │
│        └──────────────────┴──────────────────┴──────────────────┘                   │
│                                    │                                                │
│                              ALL-REDUCE                                             │
│                                    │                                                │
│                                    ▼                                                │
│                    MLP output [5 × 8192]                                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Step 3: Final LM Head (Parallel)

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│    GPU 0              GPU 1              GPU 2              GPU 3                   │
│    Vocab cols         Vocab cols         Vocab cols         Vocab cols              │
│    [0:32000]          [32000:64000]      [64000:96000]      [96000:128000]          │
│        │                  │                  │                  │                   │
│        └──────────────────┴──────────────────┴──────────────────┘                   │
│                                    │                                                │
│                              ALL-GATHER                                             │
│                                    │                                                │
│                                    ▼                                                │
│                    Logits [5 × 128000] → Sample next token                          │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Forward Pass Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     FORWARD PASS WITH TENSOR PARALLELISM                            │
│                                                                                     │
│  Input: "What is AI?" → Tokens: [1, 1724, 338, 319, 29902]                          │
│                                                                                     │
│  ═══════════════════════════════════════════════════════════════════════════════   │
│  STEP 1: Embedding Lookup                                                          │
│  ═══════════════════════════════════════════════════════════════════════════════   │
│                                                                                     │
│    Each GPU looks up its portion → ALL-GATHER → Full embeddings                    │
│                                                                                     │
│  ═══════════════════════════════════════════════════════════════════════════════   │
│  STEP 2: For each of 80 transformer layers                                         │
│  ═══════════════════════════════════════════════════════════════════════════════   │
│                                                                                     │
│    ┌─────────────────────────────────────────────────────────────────────────┐     │
│    │  LayerNorm (replicated)                                                 │     │
│    │       ↓                                                                 │     │
│    │  Attention Q,K,V (parallel by heads)                                    │     │
│    │       ↓                                                                 │     │
│    │  Attention computation (parallel)                                       │     │
│    │       ↓                                                                 │     │
│    │  Output projection (parallel) → ALL-REDUCE                              │     │
│    │       ↓                                                                 │     │
│    │  Residual connection                                                    │     │
│    │       ↓                                                                 │     │
│    │  LayerNorm (replicated)                                                 │     │
│    │       ↓                                                                 │     │
│    │  MLP Gate+Up (parallel by columns)                                      │     │
│    │       ↓                                                                 │     │
│    │  Activation (SiLU)                                                      │     │
│    │       ↓                                                                 │     │
│    │  MLP Down (parallel by rows) → ALL-REDUCE                               │     │
│    │       ↓                                                                 │     │
│    │  Residual connection                                                    │     │
│    └─────────────────────────────────────────────────────────────────────────┘     │
│                                                                                     │
│    Repeat for all 80 layers...                                                     │
│                                                                                     │
│  ═══════════════════════════════════════════════════════════════════════════════   │
│  STEP 3: Final LM Head                                                             │
│  ═══════════════════════════════════════════════════════════════════════════════   │
│                                                                                     │
│    LM Head (parallel by vocab) → ALL-GATHER → Logits → Sample next token          │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Insight

> **Every GPU processes every token through every layer** - but each only does 1/N of the computation because it only has 1/N of the weights. The ALL-REDUCE operations combine the partial results.

This is different from **pipeline parallelism** where different GPUs have different layers.


---

## Practical Sizing Guide

| Model Size | Min GPUs (24GB each) | tensor_parallel_size | Recommended Instance |
|------------|---------------------|----------------------|---------------------|
| 7-8B | 1 | 1 | 1x g5.xlarge |
| 13B | 1-2 | 1-2 | 1x g5.2xlarge or 2x g5.xlarge |
| 34B | 2-4 | 2-4 | 4x g5.xlarge |
| 70B | 4-8 | 4-8 | 8x g5.xlarge or 1x p4d.24xlarge |
| 70B (FP8) | 4 | 4 | 4x g5.xlarge |

---
