from transformers import set_seed
from accelerate import Accelerator
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import argparse
from dotenv import load_dotenv
load_dotenv()

def main(base_model: str, model_name: str):
    model = PeftModel.from_pretrained(
    AutoModel.from_pretrained(base_model, trust_remote_code=True), 
    model_id=model_name, trust_remote_code=True).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model.push_to_hub(model_name + "-merged", commit_message="Merged model")
    tokenizer.push_to_hub(model_name + "-merged", commit_message="Merged model")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    
    args = parser.parse_args()
    main(args.base_model, args.model_name)