import hydra
import wandb
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from config.train_sft_model import SFTModelTrainingConfig 
from transformers import set_seed
from dotenv import load_dotenv
from accelerate import Accelerator
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
load_dotenv()

# logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=SFTModelTrainingConfig)

@hydra.main(config_path="config/train_reward_model", version_base=None)
def main(cfg: SFTModelTrainingConfig):

    accelerator = Accelerator()

    # Configs
    model_config = cfg.model
    train_config = cfg.train
    logging_config = cfg.logging
    lora_config = model_config.lora
    data_config = cfg.dataset

    set_seed(cfg.seed)

    if logging_config.wandb and accelerator.is_main_process:
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


    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, 
                                    trust_remote_code=True, 
                                    use_cache=False)
        
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, trust_remote_code=True)

    peft_config = None
    if lora_config.enable:
        peft_config = LoraConfig(
          task_type=TaskType.SEQ_CLS,
          r = lora_config.rank,
          lora_alpha=lora_config.alpha,
          target_modules=lora_config.target_modules,
          lora_dropout=lora_config.dropout,
          bias=lora_config.bias
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    logger.info('Getting dataset')
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

    logger.info(dataset)

    # Preprocess the dataset.
    def format_data(example):
        solution = example['accepted']
        problem = example['problem']
        
        messages = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {
            "text": text,
        }
    
    dataset = dataset.map(format_data)

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            bf16=True,
            run_name=cfg.logging.wandb_run_name,
            gradient_accumulation_steps=train_config.gradient_accumulation_steps,
            gradient_checkpointing=train_config.gradient_checkpointing,
            per_device_train_batch_size=train_config.per_device_train_batch_size,
            hub_model_id=cfg.huggingface.name,
            hub_private_repo=False,
            report_to=["wandb"] if logging_config.wandb else [],
            output_dir=cfg.logging.save_dir,
            save_strategy='no',
            lr_scheduler_type=train_config.lr_scheduler_type,
            optim=train_config.optimizer,
            num_train_epochs=train_config.epochs,
            max_steps=train_config.max_steps,
            logging_steps=1,
            max_seq_length=model_config.max_length,
        ),
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Training...")
    train_results = trainer.train()
    logger.info("Training complete!")
    logger.info(train_results)

    kwargs = {
        "dataset_name": ','.join([dataset.name_or_path for dataset in cfg.dataset.datasets]),
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