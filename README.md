# eth-dl-rewards

In order to run you need the following .env file.
```
HF_TOKEN=<token>
WANDB_API_KEY=<wandb_api_key>
```

one can run preference data generation like:
```
python generate_preference_data.py --config-name="math"
```

or just login locally with Huggingface and Wandb.


One can use vast to run the script easily.
The `auto_gpu.py` script can easily be used like this for example:
```
python auto_gpu.py --run "python3 generate_preference_data.py --config-name=math"  --disk 100 --filter "gpu_name=RTX_4090 num_gpus=4 reliability>=0.99"
```
One needs to set the same variables inside the `vast.ai interface` and also login locally to `vast.ai` by adding the ssh key to the website.
You also be asked for password at some point in order to be allowed to copy the local files to remote.