from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class LoraConfig:
    enable: bool = False
    rank: int = 32
    alpha: float = 64
    target_modules: Any = "all-linear"
    dropout: float = 0.01
    bias: str = 'none'

@dataclass
class ModelConfig:
    model_name_or_path: str = "internlm/internlm2-7b-reward"
    max_length: int = 4096
    lora: LoraConfig = field(default_factory=LoraConfig)

@dataclass
class Dataset:
    name_or_path: str = "eth-dl-rewards/pref-data-math"
    split: str = "train"
    ratio: float = 1.0

@dataclass
class DatasetConfig:
    datasets: list[Dataset] = field(default_factory=list)
    max_examples: Optional[int] = None

@dataclass
class HuggingFaceConfig:
    name: str = "eth-dl-rewards/intern-lm-7b-reward-math"
    push_to_hub: bool = True

@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = "train-rm"
    wandb_run_name: str = "intern-lm-7b-reward-math-test"
    wandb_entity: str = "eth-dl-rewards"
    run_group: str = "math"
    wandb_tags: list[str] = field(default_factory=list)
    save_dir: str = "output"


@dataclass
class TrainConfig:
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    per_device_train_batch_size: int = 1
    lr_scheduler_type: str = "cosine"
    optimizer: str = "adamw_hf"
    epochs: int = 1
    max_steps: int = -1
    deepspeed_config_path: Optional[str] = None


    

@dataclass
class RewardModelTrainConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42