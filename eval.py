import hydra
import torch
import wandb
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore
from config.eval import EvalConfig 
from transformers import set_seed
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model
from datasets import load_dataset, concatenate_datasets
from trl import RewardTrainer, RewardConfig
import gc
from tqdm import tqdm
load_dotenv()

# logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=EvalConfig)

@hydra.main(config_path="config/eval", version_base=None)
def main(cfg: EvalConfig):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configs
    model_config = cfg.model
    logging_config = cfg.logging

    set_seed(cfg.seed)

    if logging_config.wandb:
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
    snapshot_download(model_config.model_name_or_path)
    

    model = AutoModel.from_pretrained(model_config.model_name_or_path, 
                                    trust_remote_code=True, 
                                    torch_dtype=torch.bfloat16,
                                    use_cache=False).to(device)
        
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, trust_remote_code=True)


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
            ).select(range(min(samples_per_dataset[i], dataset.num_rows)))

    # Concatenate the datasets
    dataset = concatenate_datasets(datasets)

    logger.info(dataset)
    batch_size = cfg.dataset.batch_size

    # Preprocess the dataset.
    def format_data(example):
        accepted = example['accepted']
        rejected = example['rejected']
        problem = example['problem']
        
        message_chosen = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": accepted}
        ]

        message_rejected = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": rejected}
        ]

        if batch_size == 1:
            message_chosen = tokenizer.apply_chat_template(message_chosen, tokenize=False)
            message_rejected = tokenizer.apply_chat_template(message_rejected, tokenize=False)
        return {
            "chosen": message_chosen,
            "rejected": message_rejected
        }
    
    dataset = dataset.map(format_data)

    chosen_texts = dataset['chosen']
    rejected_texts = dataset['rejected']

    model.eval()
    correct = 0
    total = 0

    # Calculate the number of batches
    num_batches = len(chosen_texts) // batch_size + (1 if len(chosen_texts) % batch_size != 0 else 0)

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(chosen_texts))

        # Extract the current batch of texts
        chosen_batch_texts = chosen_texts[start_idx:end_idx]
        rejected_batch_texts = rejected_texts[start_idx:end_idx]
        if batch_size == 1 or 'intern' not in model_config.model_name_or_path:
            chosen_inputs = tokenizer(chosen_texts[batch_idx], return_tensors='pt', truncation=True, max_length=4096).to(device)
            rejected_inputs = tokenizer(rejected_texts[batch_idx], return_tensors='pt', truncation=True, max_length=4096).to(device)

            with torch.no_grad():
                outputs = model(**chosen_inputs)
                reward_chosen = outputs.logits[0][0].item()
                del outputs

            with torch.no_grad():
                outputs = model(**rejected_inputs)
                reward_rejected = outputs.logits[0][0].item()
                del outputs

            if reward_chosen > reward_rejected:
                correct += 1
            total += 1
        else:

            with torch.no_grad():
                # Compute scores for the chosen and rejected batches
                scores_chosen = model.get_scores(tokenizer, chosen_batch_texts)
                scores_rejected = model.get_scores(tokenizer, rejected_batch_texts)

                # Compare rewards and update correct/total counts
                try:
                    for score_chosen, score_rejected in zip(scores_chosen, scores_rejected):
                        if score_chosen > score_rejected:
                            correct += 1
                        total += 1
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    if scores_chosen > scores_rejected:
                        correct += 1
                    total += 1

        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        if batch_idx % 10 == 0:
            logger.info(f'Processed {batch_idx * batch_size} examples with accuracy {correct/total:.4f}')
            wandb.log({'accuracy': correct / total})

    
    logger.info(f'Final accuracy: {correct/total}')
    wandb.log({'final_accuracy': correct/total})


        
    

if __name__ == "__main__":
    main()