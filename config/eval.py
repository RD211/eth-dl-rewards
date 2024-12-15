from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelConfig:
    model_name_or_path: str = "eth-dl-rewards/internlm2-7b-reward-math-30k"

@dataclass
class Dataset:
    name_or_path: str = "eth-dl-rewards/pref-data-math"
    split: str = "eval"
    ratio: float = 1.0

@dataclass
class DatasetConfig:
    datasets: list[Dataset] = field(default_factory=list)
    max_examples: Optional[int] = None

@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = "eval-rm"
    wandb_run_name: str = "internlm2-7b-reward-math-30k"
    wandb_entity: str = "eth-dl-rewards"
    run_group: str = "math"
    wandb_tags: list[str] = field(default_factory=list)
    save_every: int = 1
    save_dir: str = "output"
    save_locally: bool = True

@dataclass
class EvalConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42