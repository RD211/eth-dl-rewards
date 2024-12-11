import hydra
import wandb
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from config.train import RewardModelTrainConfig 
from transformers import set_seed
from dotenv import load_dotenv
from accelerate import Accelerator
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig
from datasets import load_dataset
from trl import RewardTrainer, RewardConfig
load_dotenv()

# logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=RewardModelTrainConfig)

@hydra.main(config_path="config/train", version_base=None)
def main(cfg: RewardModelTrainConfig):

    accelerator = Accelerator()

    # Configs
    model_config = cfg.model
    train_config = cfg.train
    logging_config = cfg.logging
    lora_config = model_config.lora
    data_config = cfg.dataset

    set_seed(cfg.seed)

    if logging_config.enable_wandb and accelerator.is_main_process:
        wandb.init(
            project=logging_config.wandb_project,
            name=logging_config.wandb_run_name,
            entity=logging_config.wandb_entity,
            group=logging_config.run_group,
            tags=logging_config.wandb_tags,
            config=OmegaConf.to_object(cfg),
        )
        include_fn = lambda x: 'output' not in x and (x.endswith('.py') or x.endswith('.yaml') or x.endswith('.txt')) and ('generate_preference_data.py' in x or 'prompt_templates' in x or 'data_generation' in x or 'config' in x)
        wandb.run.log_code('.', include_fn=include_fn)

    logger.info('Loading model')
    if accelerator.is_main_process:
        snapshot_download(model_config.model_name_or_path)
    
    accelerator.wait_for_everyone()
    model = AutoModel.from_pretrained(model_config.model_name_or_path, 
                                                 trust_remote_code=True, 
                                                 use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, trust_remote_code=True)

    peft_config = None
    if lora_config.enable:
        peft_config = LoraConfig(
          task_type="classification",
          r = lora_config.rank,
          lora_alpha=lora_config.alpha,
          target_modules=lora_config.target_modules,
          lora_dropout=lora_config.dropout,
          bias=lora_config.bias
        )

    logger.info('Getting dataset')
    dataset = load_dataset(data_config.dataset, split=data_config.split)

    # Preprocess the dataset.
    def format_data(example):
        accepted = example['accepted']
        rejected = example['rejected']
        problem = example['problem']
        
        message_chosen = {
            {"role": "user", "text": problem},
            {"role": "assistant", "text": accepted}
        }

        message_rejected = {
            {"role": "user", "text": problem},
            {"role": "assistant", "text": rejected}
        }

        return {
            "chosen": message_chosen,
            "rejected": message_rejected
        }
    
    dataset = dataset.map(format_data)

    
    
    trainer = RewardTrainer(
        model=model,
        args=RewardConfig(
            bf16=True,
            run_name=cfg.logging.run_name,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            gradient_checkpointing=train_config.gradient_checkpointing,
            per_device_train_batch_size=train_config.per_device_train_batch_size,
            hub_model_id=cfg.huggingface.name,
            hub_private_repo=False,
            report_to=["wandb"] if logging_config.wandb else [],
            output_dir=cfg.logging.save_dir,
            max_seq_length=model_config.max_model_len,
            save_strategy='no',
            lr_scheduler_type=train_config.lr_scheduler_type,
            optim=train_config.optimizer,
            num_train_epochs=train_config.epochs,
            max_steps=train_config.max_steps,
            logging_steps=1,
        ),
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Training...")
    train_results = trainer.train()
    logger.info("Training complete!")
    logger.info(train_results)

    kwargs = {
        "dataset_name": data_config.dataset,
        "tags": cfg.logging.wandb_tags,
    }

    if accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True

    if cfg.huggingface.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

        
    

if __name__ == "__main__":
    main()