import os
import hydra
import torch
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from omegaconf import OmegaConf
from datasets import load_dataset, concatenate_datasets, Dataset
from vllm import LLM, SamplingParams
from hydra.core.config_store import ConfigStore
from config.data_generation import DataGenerationConfig
from utils.preference import preference_function as PREF_FUNC
from utils.preference import Preference
from huggingface_hub import HfApi

cs = ConfigStore.instance()
cs.store(name="config", node=DataGenerationConfig)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()


@hydra.main(config_path="config/pref_data_generation", version_base=None)
def generate_preference_data(cfg: DataGenerationConfig) -> None:
  logger.info(f"Configuration: {cfg}")

  # We create output directory
  os.makedirs(cfg.logging.save_dir, exist_ok=True)

  # Do wandb logging
  if cfg.logging.wandb:
    import wandb

    wandb.init(
      project=cfg.logging.wandb_project,
      name=cfg.logging.wandb_run_name,
      entity=cfg.logging.wandb_entity,
      group=cfg.logging.run_group,
      tags=cfg.logging.wandb_tags,
      config=OmegaConf.to_object(cfg)
    )
    include_fn = lambda x: 'output' not in x and (x.endswith('.py') or x.endswith('.yaml') or x.endswith('.txt')) and ('generate_preference_data.py' in x or 'prompt_templates' in x or 'data_generation' in x or 'config' in x)
    wandb.run.log_code('.', include_fn=include_fn)

  # Load the datasets
  datasets = []
  for dataset in cfg.dataset.datasets:
    dataset = load_dataset(dataset.name_or_path, split=dataset.split)
    datasets.append(dataset)

  # We sample based on max_examples and ratios.
  if cfg.dataset.max_examples is not None:
    ratios = [dataset.ratio for dataset in cfg.dataset.datasets]
    total_ratio = sum(ratios)
    num_samples = cfg.dataset.max_examples

    # Sample based on ratios
    samples_per_dataset = [int(num_samples * ratio / total_ratio) for ratio in ratios]
    for i, dataset in enumerate(datasets):
      datasets[i] = dataset.shuffle(
        seed=cfg.seed
      ).select(range(samples_per_dataset[i]))
  
  # Concatenate the datasets
  dataset = concatenate_datasets(datasets)

  # Load vLLM model
  llm = LLM(
    model=cfg.model.model_name_or_path,
    tensor_parallel_size=torch.cuda.device_count(),
    trust_remote_code=True,
    dtype='bfloat16',
    gpu_memory_utilization=0.95,
    max_model_len=cfg.model.max_length,
    max_num_seqs=cfg.model.max_num_seqs,
  )
  
  # Load the prompt template
  with open(cfg.generation.prompt_template_path, "r") as f:
    prompt_template = f.read()

  # We apply chat template to each example in the dataset
  def apply_template(example):
    # We have a problem and solution column
    
    if cfg.dataset.ground_truth_type == "code":
      problem = example["question"]
      solution = example["input_output"]
      prompt = prompt_template.format(problem=problem)
      starter_code = example["starter_code"]
      if starter_code != "":
        prompt += f"\nHere is a starter code that you must use, we will call this function when evaluating your code:\n{starter_code}"
    else:
      problem = example["problem"]
      solution = example["solution"]
      prompt = prompt_template.format(problem=problem)

    messages = [
      {"role": "user", "content": prompt}
    ]

    return {
      "problem": problem,
      "solution": solution,
      "messages": messages
    }
  
  dataset = dataset.map(apply_template, desc="Applying chat template")

  sampling_params = SamplingParams(
    temperature=cfg.model.temperature,
    max_tokens=cfg.model.max_length,
    top_k=cfg.model.top_k,
    top_p=cfg.model.top_p
  )

  preference_function = PREF_FUNC[cfg.dataset.ground_truth_type]

  preference_data: list[tuple[str, Preference]] = []

  if cfg.huggingface.continue_from_checkpoint:
    # We load the preference data from huggingface
    checkpoint_data = load_dataset(cfg.huggingface.name, split="train")

    # We convert this into the format we need
    accepted = checkpoint_data['accepted']
    rejected = checkpoint_data['rejected']
    problem = checkpoint_data['problem']
    for row_idx in range(len(problem)):
      preference_data.append((problem[row_idx], Preference(accepted=accepted[row_idx], rejected=rejected[row_idx])))

    # We get last commit message
    commit_message = HfApi().list_repo_commits(cfg.huggingface.name, repo_type='dataset')[0].message
    last_index = int(commit_message)
    logger.info(f"Continuing from checkpoint {last_index}")
  
  index_offset = last_index + 1 if cfg.huggingface.continue_from_checkpoint else 0
    

  # Generate preferences, we batch 
  for idx in tqdm(range(index_offset, len(dataset['messages']), cfg.generation.problems_per_batch), desc="Generating preferences"):
    problems = dataset['problem'][idx:idx+cfg.generation.problems_per_batch]
    solutions = dataset['solution'][idx:idx+cfg.generation.problems_per_batch]
    messages = dataset['messages'][idx:idx+cfg.generation.problems_per_batch]

    # Generate completions
    samples = [message for message in messages for _ in range(cfg.generation.num_samples_per_problem)]
    completions = llm.chat(samples, sampling_params=sampling_params)
    text_completions = [completion.outputs[0].text for completion in completions]
    
    # Compute preferences
    preferences: list[tuple[str, Preference]] = []

    for i in range(len(problems)):
      problem = problems[i]
      solution = solutions[i]
      responses = text_completions[i*cfg.generation.num_samples_per_problem:(i+1)*cfg.generation.num_samples_per_problem]
      pref_gen = preference_function(responses, solution, cfg.preference.max_number_of_preferences_per_problem)
      preferences.extend([(problem, pref) for pref in pref_gen])

    # Add to preference data
    preference_data.extend(preferences)

    
    # Save the preference data
    if idx % cfg.logging.save_every == 0 or idx >= len(dataset['messages']) - cfg.generation.problems_per_batch:
      
      logger.info(f"Saving preference data for batch {idx}")
      transformed_preference_data = []
      for problem, pref in preference_data:
        transformed_preference_data.append({
          "problem": problem,
          "accepted": pref.accepted,
          "rejected": pref.rejected
        })

      # 3 Columns problem, accepted, rejected
      df = pd.DataFrame(transformed_preference_data)

      if cfg.logging.save_locally:
        df.to_csv(os.path.join(cfg.logging.save_dir, f"preference_data_{idx}_{time.time()}_{cfg.logging.wandb_run_name}.csv"))
      

      # Push to huggingface
      if cfg.huggingface.push_to_hub:
        dataset_to_save = Dataset.from_pandas(df)
        dataset_to_save.push_to_hub(
          cfg.huggingface.name,
          commit_message=cfg.huggingface.commit_message + f" {idx + cfg.generation.problems_per_batch}",
          commit_description=str(idx + cfg.generation.problems_per_batch)
        )
        
    




if __name__ == "__main__":
  generate_preference_data()