# Can Reward Models Transfer Across Domains in Large Language Models?


## Overview
[Link to paper](paper.pdf)

Reinforcement Learning from Human Feedback (RLHF) aligns language models (LMs) with human preferences by training reward models on preference data and optimizing LMs to maximize these rewards. However, obtaining effective reward models is resource-intensive and typically requires separate models for different tasks, leading to data inefficiencies. This repository investigates the transferability of reward models across domains in Large Language Models (LLMs) through fine-tuning.

We hypothesize that fundamental logic can be captured by transferable reward models, which can be fine-tuned with minimal data for specific tasks. Additionally, we integrate an **Active Learning framework** to enhance data efficiency by iteratively selecting the most uncertain samples for fine-tuning. Our empirical investigations in mathematics and coding domains confirm the transferability of general-purpose reward models and the potential of cross-domain adaptability. Our findings also reveal that the performance of cross-domain adaptation depends on the source and target domains, hinting at the asymmetric nature of the inherent characteristics of different domains. 


## Pre-trained Models and Datasets 🤗

- **Pre-trained Models**: All models are available on [Hugging Face 🤗](https://huggingface.co/eth-dl-rewards):
  - [Math Reward Models](https://huggingface.co/eth-dl-rewards?search_models=math)
  - [Code Reward Models](https://huggingface.co/eth-dl-rewards?search_models=code)
  - [Math SFT Model](https://huggingface.co/eth-dl-rewards/internlm2-7b-sft-math)
  - [Code SFT Model](https://huggingface.co/eth-dl-rewards/internlm2-7b-sft-code)

- **Datasets**: Access the generated preference datasets:
  - [Math Preference Data](https://huggingface.co/datasets/eth-dl-rewards/pref-data-math)
  - [Code Preference Data](https://huggingface.co/datasets/eth-dl-rewards/pref-data-code)

---

## Getting Started

### Prerequisites

Create a `.env` file with the following variables:

```bash
HF_TOKEN=<your_huggingface_token>
WANDB_API_KEY=<your_wandb_api_key>
```

#### Docker Image

Use the `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` Docker image, and set up the environment by running:

```bash
./setup.sh
```

All experiments are conducted using an H200 GPU from [Vast.AI](https://vast.ai).

#### Auto GPU

We provide an automated script that we made to streamline the setup and GPU allocation process via `Vast.ai`. Refer to [AUTO_GPU](auto_gpu.md) for details.

---

## Generating Preference Data

To generate preference data, use:

```bash
python generate_preference_data.py --config-name="math" # or "code"
```

- For **math**, answers are validated using the `latex2sympy` library.
- For **code**, correctness is verified by running test cases.

Preference pairs are created by matching incorrect or lower-quality solutions with correct or higher-quality ones.

---

## Training the Reward Models

Train reward models using `Accelerate` and `Deepspeed`. Example:

```bash
accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=math_scratch.yaml 
```

Our training leverages **LoRA** (Low-Rank Adaptation) with a rank of 32. Increasing the rank did not yield additional performance improvements.

---

## Evaluating the Models

Evaluate the reward model by measuring its ability to assign a higher reward to preferred answers:

```bash
accelerate launch --config_file=deepspeed/1_1GPU.yaml eval.py --config-name=math_scratch.yaml 
```

---

## Active Learning

Active learning enhances data efficiency by iteratively selecting the most uncertain samples for fine-tuning. The workflow consists of three scripts, executed in sequence, repeated for 3 iterations with 2.5k samples selected in each iteration:

1. **Data Selection**: Identify uncertain samples for training.

    ```bash
    accelerate launch --config_file=deepspeed/1_1GPU.yaml active_data_selection.py --config-name=math.yaml
    ```

2. **Fine-Tuning**: Train the reward model using the selected data.

    ```bash
    accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=active_learning_math.yaml
    ```

3. **Evaluation**: Assess the performance of the fine-tuned model.

    ```bash
    accelerate launch --config_file=deepspeed/1_1GPU.yaml eval.py --config-name=active_learning_math.yaml
    ```

    After evaluation, merge the fine-tuned model with the base model:

    ```bash
    python3 merge_peft_model.py --base_model internlm/internlm2-7b-reward --model_name eth-dl-rewards/internlm2-7b-reward-math-active
    ```

This process is repeated for 3 iterations, ensuring the model focuses on the most informative samples, minimizing labeling costs, and improving performance.

**Note**: For the first iteration of active learning, use the scripts with `_initial` appended to their names. All subsequent calls should use the standard versions.


---

## Authors

- Changling Li  
- David Dinucu-Jianu  
- Michelle Chen  
- Yuan Gao  

---