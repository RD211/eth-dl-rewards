from transformers import AutoModel, AutoTokenizer, AutoConfig
from peft import PeftModel
import argparse
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi

load_dotenv()

def main(base_model: str, model_name: str, reward_model: str = "internlm/internlm2-7b-reward"):
    model = PeftModel.from_pretrained(
    AutoModel.from_pretrained(base_model, trust_remote_code=True), 
    model_id=model_name, trust_remote_code=True).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model.push_to_hub(model_name + "-merged", commit_message="Merged model")
    tokenizer.push_to_hub(model_name + "-merged", commit_message="Merged model")
    
    
    path = hf_hub_download(repo_id=reward_model, filename="modeling_internlm2.py")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo="modeling_internlm2.py",
        repo_id=model_name + "-merged",
        repo_type="model",
    )

    path = hf_hub_download(repo_id=reward_model, filename="config.json")
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo="config.json",
        repo_id=model_name + "-merged",
        repo_type="model",
    )

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    
    args = parser.parse_args()
    main(args.base_model, args.model_name)