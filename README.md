# Recursive Language Model - 90M Parameters

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗-Model%20Card-yellow)](https://huggingface.co/Girinath11/recursive-language-model-90m)

A novel 90M parameter conversational language model featuring **adaptive recursive computation** through perplexity-based dynamic routing. Achieves **17.45 perplexity** — better than GPT-2 Small (29 perplexity) with 30% fewer parameters.

<div align="center">


**[🤗 Model Card](https://huggingface.co/Girinath11/recursive-language-model-90m)** • **[📖 Blog Post](https://medium.com/@girinathv)** 

</div>

---

## 🏆 Key Achievement
```
Perplexity: 17.45 (Validation Set)
Outperforms GPT-2 Small (117M params, ~29 perplexity)
90.7M parameters (30% smaller than GPT-2 Small)
```

## 🔥 Innovation: Self-Supervised Adaptive Computation

Unlike traditional transformers that apply uniform computation to all inputs, this model **learns to allocate computation based on difficulty**:
```python
# Model learns from its own perplexity signals:
High Perplexity (>50)  → Model struggling → 5 recursion steps
Medium (20-50)         → Moderate        → 3 steps
Low Perplexity (<20)   → Confident       → 1 step (fast!)
```

**Key Innovation:** Zero manual labeling — the model learns what's "hard" from its own confidence signals!

---

## 📊 Performance

### Benchmark Comparison

| Model | Parameters | Perplexity | Notes |
|-------|-----------|------------|-------|
| **This Model** | **90.7M** | **17.45** | ✅ Novel adaptive architecture |
| GPT-2 Small | 117M | ~29 | Baseline comparison |
| GPT-2 Medium | 345M | ~22 | 3.8× larger |
| DistilGPT-2 | 82M | ~35 | Distilled model |

### Training Progression
```
Epoch 1: 35.39 perplexity
Epoch 2: 20.27 perplexity (43% improvement)
Epoch 3: 17.45 perplexity (51% total improvement) 🔥

Training Time: 84 minutes on single T4 GPU
Loss Reduction: 4.50 → 2.86 (36% improvement)
```

<details>
<summary>📈 View Training Curves</summary>

![Training Progress](./assets/training_progress.png)

</details>

---

## 🎯 Model Architecture

### Specifications

| Component | Configuration |
|-----------|--------------|
| **Total Parameters** | 90,697,603 (~90.7M) |
| **Vocabulary** | 50,259 tokens (GPT-2 BPE + special tokens) |
| **Embedding Dim** | 560 |
| **Layers** | 8 transformer layers |
| **Attention Heads** | 8 heads × 70 dim |
| **FFN Size** | 2240 |
| **Max Length** | 512 tokens |
| **Positional Encoding** | RoPE (Rotary Position Embeddings) |

### Architecture Components
```
Input → Token Embeddings (28.1M params)
     → Base Transformer Stack (8 layers, 30.5M params)
     → Perplexity-Based Router (0.4M params)
          ├─ Simple (PPL<20)   → 1 recursion step
          ├─ Medium (PPL 20-50) → 3 steps
          └─ Complex (PPL>50)   → 5 steps
     → Recursive Layer (3.8M params, applied 1-5×)
     → Final Norm + LM Head
     → Output Predictions
```

**Innovation:** Router learns complexity from model's own perplexity — no manual labels!

---

## 🚀 Quick Start

### Installation
```bash
pip install transformers torch
```

### Basic Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Girinath11/recursive-language-model-90m",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "Girinath11/recursive-language-model-90m"
)

# Move to device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# Generate
prompt = "<|user|>\nWhat is machine learning?\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

print(tokenizer.decode(outputs[0]))
```

### Conversational Format
```python
# Model expects this format:
conversation = """<|user|>
Explain quantum computing simply.
<|assistant|>
"""

# Generate response
inputs = tokenizer(conversation, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.8)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

<details>
<summary>🔧 Advanced Usage Examples</summary>

**Batch Generation:**
```python
prompts = [
    "<|user|>\nWrite a haiku about AI.\n<|assistant|>\n",
    "<|user|>\nExplain neural networks.\n<|assistant|>\n"
]

inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
outputs = model.generate(**inputs, max_new_tokens=80)

for i, output in enumerate(outputs):
    print(f"Response {i+1}:", tokenizer.decode(output))
```

**Temperature Control:**
```python
# Creative (high temperature)
creative = model.generate(**inputs, temperature=1.0, top_p=0.95)

# Focused (low temperature)  
focused = model.generate(**inputs, temperature=0.5, top_p=0.9)

# Greedy (deterministic)
greedy = model.generate(**inputs, do_sample=False)
```

</details>

---

## 📚 Training Details

### Dataset

**Total:** 37,119 premium conversational samples

| Dataset | Source | Samples | Quality |
|---------|--------|---------|---------|
| Anthropic HH-RLHF | Anthropic | 30,000 | Claude-style responses |
| UltraChat | Tsinghua | 7,119 | GPT-4 conversations |
| **Validation** | Mixed | 2,000 | Held-out test set |

**Quality Filtering:**
- ✅ Minimum 80 characters, maximum 6,000
- ✅ Minimum 15 words with proper punctuation
- ✅ Alphanumeric ratio > 65%
- ✅ Valid conversation structure

### Training Configuration
```yaml
Hardware:
  GPU: NVIDIA T4 (15GB)
  Platform: Kaggle
  Precision: FP16 (Mixed Precision)

Hyperparameters:
  Batch Size: 8
  Gradient Accumulation: 8
  Effective Batch: 64
  Learning Rate: 3e-4
  Optimizer: AdamW
  Schedule: OneCycleLR
  Epochs: 3
  Total Steps: 13,917

Loss:
  Primary: Language Modeling (CrossEntropy)
  Auxiliary: Router Loss (0.1 weight)
  Total: LM Loss + 0.1 × Router Loss
```

**Training Time:** 84 minutes (~1.4 hours)

<details>
<summary>📊 Detailed Training Metrics</summary>

| Epoch | Train Loss | Eval Loss | Perplexity | Time |
|-------|-----------|-----------|------------|------|
| 1 | 4.4979 | 3.5665 | 35.39 | 29 min |
| 2 | 3.2960 | 3.0091 | 20.27 | 28 min |
| 3 | 2.8572 | 2.8595 | **17.45** 🔥 | 27 min |

**Key Observations:**
- ✅ Steady convergence across all epochs
- ✅ No overfitting (train/eval losses aligned)
- ✅ 51% perplexity improvement
- ✅ Stable training speed (~2.7 it/s)

</details>

---

## 💡 Technical Innovation

### Perplexity-Based Routing (Novel Contribution)

**Problem:** Traditional models apply the same depth to all inputs.

**Solution:** Learn complexity from model's own performance!
```python
# During training:
sample_perplexity = exp(sample_loss)

# Auto-generate pseudo-labels:
if sample_perplexity < 20:
    complexity = "simple"    # 1 recursion step
elif sample_perplexity < 50:
    complexity = "medium"    # 3 steps
else:
    complexity = "complex"   # 5 steps

# Router learns to predict this
router_loss = CrossEntropyLoss(router_logits, complexity)
```

**Benefits:**
- ✅ No manual labeling required
- ✅ Adapts as model learns (curriculum learning)
- ✅ Objective difficulty measure
- ✅ Efficient compute allocation

### Self-Supervised Curriculum Learning

The model implements automatic curriculum:
```
Early Training:  Most samples "complex" (high perplexity)
Mid Training:    Distribution shifts as model learns
Late Training:   More samples become "simple" (low perplexity)
```

This creates natural curriculum without human intervention!

---

## 🎯 Use Cases

### ✅ Recommended

- 🎓 **Educational demos** - Teaching language model concepts
- 🔬 **Research** - Experimenting with adaptive computation
- 🛠️ **Prototyping** - Testing conversational AI applications
- 💡 **Learning** - Understanding transformer architectures
- ✍️ **Creative assistance** - With human review

### ⚠️ Not Recommended

- ❌ Production chatbots without human oversight
- ❌ Medical, legal, or financial advice
- ❌ Automated content moderation
- ❌ Safety-critical systems
- ❌ High-stakes decision making

---

## ⚠️ Limitations

### Technical Constraints

- **Context Window:** 512 tokens (vs 2048+ for modern models)
- **Training Data:** 37K samples (relatively small)
- **Language:** Primarily English
- **Knowledge Cutoff:** Early 2024

### Known Issues

- May repeat phrases in long generations (>200 tokens)
- Limited factual knowledge (small training set)
- May lose context in very long conversations
- Best with clear, specific prompts

---
---

## 🔧 Development

### Setup
```bash
# Clone repository
git clone https://github.com/Girinath11/recursive-language-model.git
cd recursive-language-model

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

### Training from Scratch
```bash
# Train model
python training/train.py \
  --batch_size 8 \
  --epochs 3 \
  --lr 3e-4 \
  --output_dir ./checkpoints

# Evaluate
python training/evaluate.py \
  --model_path ./checkpoints/best_model
```

---

## 📖 Citation

If you use this model in your research:
```bibtex
@misc{recursive-lm-90m-2026,
  author = {Girinath V},
  title = {Recursive Language Model with Perplexity-Based Dynamic Routing},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/Girinath11/recursive-language-model-90m}},
  note = {90M parameter model achieving 17.45 perplexity}
}
```

---

## 🙏 Acknowledgments

- **Anthropic** for HH-RLHF dataset
- **Tsinghua University** for UltraChat dataset
- **Hugging Face** for Transformers library
- **Kaggle** for free GPU access
- **OpenAI** for GPT-2 tokenizer

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details

---

## 📧 Contact

**Girinath V**

- 💼 [LinkedIn](https://linkedin.com/in/girinathv)
- 🐙 [GitHub](https://github.com/Giri530)
- 📧 girinathv48@gmail.com
- 🤗 [Hugging Face](https://huggingface.co/Girinath11)

---

<div align="center">

**Made with ❤️ and lots of ☕**

**[⬆ Back to Top](#recursive-language-model---90m-parameters)**

</div>
