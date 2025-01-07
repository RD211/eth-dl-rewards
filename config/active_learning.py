from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    model_name_or_path: str = "eth-dl-rewards/internlm2-7b-reward-math-30k"

@dataclass
class DatasetConfig:
    dataset_name_or_path: str = "eth-dl-rewards/active-learning-math-data-initial"
    output_dataset_name_or_path: str = "eth-dl-rewards/active-learning-data"
    pairs_to_select: int = 1000
    batch_size: int = 8
    max_to_look_at: int = 40000

@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = "active-learning-data"
    wandb_run_name: str = "internlm2-7b-reward-math-30k"
    wandb_entity: str = "eth-dl-rewards"
    run_group: str = "math"
    wandb_tags: list[str] = field(default_factory=list)
    save_every: int = 1
    save_dir: str = "output"
    save_locally: bool = True


@dataclass
class ActiveLearningConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42