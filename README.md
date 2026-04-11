# GPT-2 from Scratch: Decoder-Only Transformer

This repository contains a complete, from-scratch implementation of a **GPT-style Decoder-only Transformer**. The project covers the entire pipeline from raw data ingestion and sub-word tokenization to a functional training loop.

## 🚀 Technical Approach

The model is designed as a generative language model that leverages self-attention to understand and predict text sequences.

### 1. Data & Tokenization
* **Custom BPE Tokenizer:** Implemented a **Byte Pair Encoding (BPE)** algorithm from scratch. This sub-word level tokenizer allows the model to handle a large vocabulary efficiently while managing rare words better than character-level models.
* **Embeddings:** Combines **Contextual Embeddings** (learned during training) with **Positional Embeddings** to ensure the model understands the relative order of tokens.
* **Pre-Normalization:** Uses **Layer Normalization** at the start of each transformer block (Pre-Norm) to ensure training stability and faster convergence.

### 2. Decoder-Only Architecture
* **Multi-Head Attention (MHA):** Implemented multiple attention heads to allow the model to focus on different parts of the sequence simultaneously (capturing various linguistic perspectives).
* **Residual Connections:** Every block uses skip-connections to ensure gradients flow easily through deep layers.
* **Feed-Forward Network (FFN):** Includes a linear bottleneck that expands the hidden dimension by **4x** to capture "factual" knowledge before projecting back to the embedding size.
* **Softmax Output:** The final layer produces a probability distribution across the entire BPE vocabulary to predict the next token.

---

## 📂 Codebase Structure

The project is organized into modular files to separate the data logic from the model architecture:

| File | Description |
| :--- | :--- |
| **`Dataloader.py`** | Handles remote data fetching and raw text stream management. |
| **`BPETokeniser.py`** | The scratch-built BPE logic for training and encoding/decoding. |
| **`Preprocessing.py`** | Manages `get_batch` logic, tensor conversion, and train/val splitting. |
| **`Block.py`** | Contains the Transformer layers, Multi-Head Attention, and FFN modules. |
| **`train.py`** | The main entry point. Contains the training loop and generation logic. |

---

## 📊 Sample Outputs

### Training Logs & Generation Sequence
Below is the sequential progression of the model's text generation (predicting Shakespearean-style dialogue) as captured during the training process:

#### Initial Output (Sequence Part 1)
![Generation Output Part 1](output1.png)

#### Progression (Sequence Part 2)
![Generation Output Part 2](output2(continued).png)

#### Advanced Context (Sequence Part 3)
![Generation Output Part 3](output3(continued).png)

#### Refined Dialogue (Sequence Part 4)
![Generation Output Part 4](output4(continued).png)

---
