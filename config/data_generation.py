from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    model_name_or_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    temperature: float = 0.0
    max_length: int = 512
    top_k: int = 50
    top_p: float = 1.0
    max_num_seqs: int = 8

@dataclass
class GenerationConfig:
    num_samples_per_problem: int = 8
    prompt_template_path: str = "prompt_templates/math.txt"
    problems_per_batch: int = 8

@dataclass
class Dataset:
    name_or_path: str = "AI-MO/NuminaMath-CoT"
    split: str = "train"
    ratio: float = 1.0

@dataclass
class DatasetConfig:
    datasets: list[Dataset] = field(default_factory=list)
    max_examples: Optional[int] = None
    ground_truth_type: str = "math" # math / mcq / freeform

@dataclass
class HuggingFaceConfig:
    name: str = "eth-dl-rewards/pref-data-math"
    push_to_hub: bool = True
    commit_message: str = "Add math preferences"
    continue_from_checkpoint: bool = False

@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = "pref-data"
    wandb_run_name: str = "test"
    wandb_entity: str = "eth-dl-rewards"
    run_group: str = "math"
    wandb_tags: list[str] = field(default_factory=list)
    save_every: int = 1
    save_dir: str = "output"
    save_locally: bool = True

@dataclass
class PreferenceConfig:
    max_number_of_preferences_per_problem: int = 4

@dataclass
class DataGenerationConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    preference: PreferenceConfig = field(default_factory=PreferenceConfig)
    seed: int = 42