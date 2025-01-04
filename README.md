# eth-dl-rewards

In order to run you need the following .env file.

```bash
HF_TOKEN=<token>
WANDB_API_KEY=<wandb_api_key>
```


## Generate preference data
one can run preference data generation like:

```bash
python generate_preference_data.py --config-name="math"
```

or just login locally with Huggingface and Wandb.


One can use vast to run the script easily.
The `auto_gpu.py` script can easily be used like this for example:

```bash
python auto_gpu.py --run "python3 generate_preference_data.py --config-name=math model.max_num_seqs=128"  --disk 100 --filter "gpu_name=RTX_4090 num_gpus=4 reliability>=0.99"
```

One needs to set the same variables inside the `vast.ai interface` and also login locally to `vast.ai` by adding the ssh key to the website.
You also be asked for password at some point in order to be allowed to copy the local files to remote.


## Train

```bash
accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=math
```


```bash
python auto_gpu.py --run "accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=math"  --disk 100 --filter "gpu_name=H100_SXM num_gpus=1 reliability>=0.99"
```

Also works with multiple gpus just select `3_4GPU.yaml` for 4 GPUs or change the config.

## Eval

```
python auto_gpu.py --run "accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=math_0k.yaml;accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=math_30k.yaml;accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=math_60k.yaml;accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=math_100k.yaml" --disk 100 --filter "gpu_name=L40S num_gpus=1 reliability>=0.99"
```

python auto_gpu.py --run "accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=math_scratch.yaml" --disk 100 --filter "gpu_name=L40S num_gpus=1 reliability>=0.99"



# Reproduce scripts
Trains all code models.

```bash
python auto_gpu.py --run "
accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=code_20k; 
accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=code_40k; 
accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=code_60k; 
accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=code_scratch"  --disk 100 --filter "gpu_name=H200 num_gpus=1"
```

Eval all code models.

```bash
python auto_gpu.py --run "accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=code_0k.yaml;
accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=code_20k.yaml;
accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=code_40k.yaml;
accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=code_60k.yaml; 
accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=code_scratch.yaml" --disk 100 --filter "gpu_name=H100_SXM num_gpus=1 reliability>=0.99" 
```

Trains all math models.

```bash
python auto_gpu.py --run "
accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=math_20k; 
accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=math_40k; 
accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=math_60k; 
accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=math_scratch"  --disk 100 --filter "gpu_name=H200 num_gpus=1"
```

Eval all math models.

```bash
python auto_gpu.py --run "accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=math_0k.yaml;
accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=math_20k.yaml;
accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=math_40k.yaml;
accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=math_60k.yaml; 
accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=math_scratch.yaml" --disk 100 --filter "gpu_name=H100_SXM num_gpus=1 reliability>=0.99" 
```

Merge scratch models.

```bash
python auto_gpu.py --run "
python3 merge_peft_model.py --base_model eth-dl-rewards/internlm2-7b-mod --model_name eth-dl-rewards/internlm2-7b-reward-code-60k-scratch;
python3 merge_peft_model.py --base_model eth-dl-rewards/internlm2-7b-mod --model_name eth-dl-rewards/internlm2-7b-reward-math-60k-scratch
" --disk 100 --filter "gpu_name=H100_SXM num_gpus=1 reliability>=0.99" 
```

(the config and modelling file must also be copied to do this step)

Train cross-domain code -> math.

```bash
python auto_gpu.py --run "accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=code_to_math_20k"  --disk 100 --filter "gpu_name=H200 num_gpus=1" --auto --detach ;
python auto_gpu.py --run "accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=code_to_math_40k"  --disk 100 --filter "gpu_name=H200 num_gpus=1" --auto --detach ;
python auto_gpu.py --run "accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=code_to_math_60k"  --disk 100 --filter "gpu_name=H200 num_gpus=1" --auto --detach
```

Eval cross domain code -> math.

```bash
python auto_gpu.py --run "accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=code_to_math_20k.yaml;
accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=code_to_math_40k.yaml;
accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=code_to_math_60k.yaml" --disk 100 --filter "gpu_name=H100_SXM num_gpus=1 reliability>=0.99" 
```

Train cross-domain math -> code.

```bash
python auto_gpu.py --run "accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=math_to_code_20k"  --disk 100 --filter "gpu_name=H200 num_gpus=1" --auto --detach ;
python auto_gpu.py --run "accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=math_to_code_40k"  --disk 100 --filter "gpu_name=H200 num_gpus=1" --auto --detach ;
python auto_gpu.py --run "accelerate launch --config_file=deepspeed/1_1GPU.yaml train_reward_model.py --config-name=math_to_code_60k"  --disk 100 --filter "gpu_name=H200 num_gpus=1" --auto --detach
```

Eval cross domain math -> code.

```bash
python auto_gpu.py --run "accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=math_to_code_20k.yaml;
accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=math_to_code_40k.yaml;
accelerate launch --config_file deepspeed/1_1GPU.yaml eval.py --config-name=math_to_code_60k.yaml" --disk 100 --filter "gpu_name=H100_SXM num_gpus=1 reliability>=0.99" 
```