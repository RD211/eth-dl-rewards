from config.active_learning import ActiveLearningConfig
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
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from trl import RewardTrainer, RewardConfig
import numpy as np
import gc
from tqdm import tqdm
load_dotenv()

# logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=EvalConfig)

@hydra.main(config_path="config/active_learning", version_base=None)
def main(cfg: ActiveLearningConfig):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configs
    model_config = cfg.model
    logging_config = cfg.logging
    dataset_config = cfg.dataset

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
    rest_dataset = None
    if 'initial' in dataset_config.dataset_name_or_path:
        dataset = load_dataset(dataset_config.dataset_name_or_path, split='train')
        train_dataset = dataset
    else:
        train_dataset = None
        rest_dataset = None
        
        train_dataset = load_dataset(dataset_config.dataset_name_or_path, split='train')
        rest_dataset = load_dataset(dataset_config.dataset_name_or_path, split='rest')
     
    # If train is the only dataset, use it
    if train_dataset is not None and rest_dataset is None:
        dataset = train_dataset
        
    # If rest exists, use it and we concat train with used.
    if rest_dataset is not None:
        dataset = rest_dataset
    

    logger.info(dataset)

    # Preprocess the dataset.
    def format_data(example):
        accepted = example['accepted']
        rejected = example['rejected']
        problem = example['problem']
        
        message_accepted = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": accepted}
        ]

        message_rejected = [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": rejected}
        ]

        return {
            "problem": problem,
            "accepted": message_accepted,
            "rejected": message_rejected,
            "original_accepted": accepted,
            "original_rejected": rejected
        }
    
    dataset = dataset.map(format_data)
    dataset = dataset.shuffle(seed=cfg.seed)
    dataset = dataset.select(range(min(len(dataset['problem']), dataset_config.max_to_look_at)))

    problems = dataset['problem']
    accepted_texts = dataset['accepted']
    rejected_texts = dataset['rejected']

    model.eval()
    
    gaps = []
    batch_size = dataset_config.batch_size
    
    # Process the inputs in batches
    num_batches = len(accepted_texts) // batch_size + (1 if len(accepted_texts) % batch_size != 0 else 0)

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(accepted_texts))

        # Extract the current batch of texts
        accepted_batch_texts = accepted_texts[start_idx:end_idx]
        rejected_batch_texts = rejected_texts[start_idx:end_idx]

        with torch.no_grad():
            # Compute model outputs for the accepted and rejected batches
            outputs_accepted = model.get_scores(tokenizer, accepted_batch_texts)
            outputs_rejected = model.get_scores(tokenizer, rejected_batch_texts)

            # Compute the absolute differences and extend the gaps list
            gaps.extend([abs(accepted - rejected) for accepted, rejected in zip(outputs_accepted, outputs_rejected)])

        if batch_idx % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    original_accepted_texts = dataset['original_accepted']
    original_rejected_texts = dataset['original_rejected']
    pairs_to_select = dataset_config.pairs_to_select
    
    # We take the pairs with the lowest gaps.

    def sample_by_gap(gaps, k, temp=1.0):
        w = np.exp(-np.array(gaps)/temp)
        return np.random.choice(len(gaps), k, replace=False, p=w/w.sum())

    indices = sample_by_gap(gaps, pairs_to_select)
    selected_problems = [problems[i] for i in indices]
    selected_accepted_texts = [original_accepted_texts[i] for i in indices]
    selected_rejected_texts = [original_rejected_texts[i] for i in indices]
    
    # we take all not selected pairs and put them in other dataset
    indices = set(list(indices))
    rest_indices = [i for i in range(len(gaps)) if i not in indices]
    rest_problems = [problems[i] for i in rest_indices]
    rest_accepted_texts = [original_accepted_texts[i] for i in rest_indices]
    rest_rejected_texts = [original_rejected_texts[i] for i in rest_indices]
    
    # We create the new dataset
    rest_dataset = []
    for i in range(len(rest_problems)):
        rest_dataset.append({
            'problem': rest_problems[i],
            'accepted': rest_accepted_texts[i],
            'rejected': rest_rejected_texts[i]
        })
    rest_dataset = Dataset.from_list(rest_dataset)
    # We create the new train dataset
    train_dataset = []
    for i in range(len(selected_problems)):
        train_dataset.append({
            'problem': selected_problems[i],
            'accepted': selected_accepted_texts[i],
            'rejected': selected_rejected_texts[i]
        })
    train_dataset = Dataset.from_list(train_dataset)

    dataset = DatasetDict({
        'train': train_dataset,
        'rest': rest_dataset,
    })
    
    dataset.push_to_hub(dataset_config.output_dataset_name_or_path)

        
    

if __name__ == "__main__":
    main()