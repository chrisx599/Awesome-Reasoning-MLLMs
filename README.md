# Awesome Reasoning in Multi-modal Large Language Models (MLLMs)

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A curated list of resources for improving reasoning capabilities in Multi-modal Large Language Models (MLLMs).  
*Contributions welcome!* 🚀

---

## 📖 Introduction
This repository focuses on cutting-edge techniques, datasets, and tools to enhance ​**reasoning abilities** (logical, mathematical, commonsense, and multimodal reasoning) in MLLMs.  
**Key Features**:
- 🧠 Covers ​**text, image, and video** modalities
- 📊 Benchmarks and evaluation metrics
- 🔧 Ready-to-use code examples

---

## 🗂️ Table of Contents
- [Papers & Models](#-papers--models)
- [Datasets](#-datasets)
- [Tools & Libraries](#-tools--libraries)
- [Tutorials & Examples](#-tutorials--examples)
- [Evaluation](#-evaluation)
- [Contribution Guidelines](#-contribution-guidelines)

---

## 📚 Papers & Models
### Reasoning Techniques
| Category          | Key Papers                                                                 | Models/Code                                                                 |
|-------------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| ​**Chain-of-Thought** | [Chain-of-Thought Prompting (Wei et al., 2022)](https://arxiv.org/abs/2201.11903) | [LLaMA-2 CoT](https://github.com/facebookresearch/llama)                   |
| ​**Self-Consistency** | [Self-Consistency Improves CoT (Wang et al., 2023)](https://arxiv.org/abs/2203.11171) | [Google/FLAN-T5](https://huggingface.co/google/flan-t5-xxl)                |
| ​**Tool-Augmented**  | [Toolformer (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)      | [Toolformer Demo](https://github.com/lucidrains/toolformer-pytorch)        |

### Multimodal Reasoning Models
| Model       | Modality   | Highlights                                | Code                                                                 |
|-------------|------------|-------------------------------------------|----------------------------------------------------------------------|
| ​**LLaVA**   | Image+Text | Visual instruction tuning                 | [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)                |
| ​**Flamingo**| Image+Text | Few-shot visual reasoning                 | [Flamingo Demo](https://huggingface.co/docs/transformers/model_doc/flamingo) |
| ​**CogVLM**  | Image+Text | Granular vision-language alignment        | [CogVLM GitHub](https://github.com/THUDM/CogVLM)                    |

---

## 📊 Datasets
| Dataset     | Modality   | Task Type                | Link                                                                 |
|-------------|------------|--------------------------|----------------------------------------------------------------------|
| ​**GSM8K**   | Text       | Math Word Problems       | [GSM8K Dataset](https://github.com/openai/grade-school-math)        |
| ​**ScienceQA**| Image+Text | Multimodal QA           | [ScienceQA GitHub](https://scienceqa.github.io/)                    |
| ​**FOLIO**   | Text       | Logical Reasoning        | [FOLIO GitHub](https://github.com/Yale-LILY/FOLIO)                  |

---

## 🔧 Tools & Libraries
- ​**Chain-of-Thought Generators**:
  - [LangChain CoT Module](https://python.langchain.com/docs/modules/chains/foundational/thought_generation)
  - [Vicuna Prompt Template](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py)
- ​**Multimodal Evaluation**:
  - [VQA Evaluation Kit](https://github.com/GT-Vision-Lab/VQA)
  - [CIDEr Metric](https://github.com/tylin/coco-caption)

---

## 🧪 Tutorials & Examples
### Basic Usage
```python
# Example: Loading LLaVA for visual reasoning
from transformers import LlavaForConditionalGeneration, AutoProcessor

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
# See full example in examples/visual_reasoning.ipynb
```

### Advanced Guides
Fine-tuning MLLMs for Math Reasoning
Tool-Augmented Reasoning with Python

## 📈 Evaluation
Leaderboard (MATH Dataset)
Model	Accuracy	Paper
GPT-4	92.5%	OpenAI (2023)
LLEMMA-34B	85.2%	LLEMMA Paper

## 🤝 Contribution Guidelines
​Add a Paper: Submit a PR with a link to the paper and code.
​Fix Errors: Open an Issue with "bug" label.
​Formatting: Use markdown tables for papers/datasets.
See CONTRIBUTING.md for details.

## 📜 License
This project is licensed under the MIT License.
Please cite original papers when using their code/data.

