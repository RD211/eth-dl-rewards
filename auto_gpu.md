# Auto GPU Selection Script

## Overview
This script automates the process of selecting and setting up a GPU instance on Vast.ai, including transferring files, running commands, and cleaning up resources.

## Prerequisites
1. Ensure your **SSH key** is added to [Vast.ai](https://vast.ai).
2. Install the required dependencies:
   - Python 3.8+
   - `rich`
   - `curses` (comes pre-installed with Python)

3. Have the **Vast.ai API key** stored in `~/.vast_api_key` on your local machine.

4. Prepare the `.auto_gpu` directory to store temporary files created during the process:
   ```bash
   mkdir -p .auto_gpu
   ```

## Usage

### Running the Script
```bash
python auto_gpu.py --run "<command-to-run>" [options]
```

### Options
- `--filter`: Filter string for Vast.ai GPU search.
- `--run`: The command to execute on the selected GPU instance.
- `--disk`: Required disk capacity in GB (will prompt if not provided).
- `--auto`: Automatically select the first available GPU without user interaction.
- `--image`: Docker image to use on the GPU instance (default: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`).
- `--env`: Environment variables for the Docker container.
- `--onstart`: Command to execute on instance startup (default: prepare environment).
- `--keep`: Keep the instance running after execution.
- `--detach`: Run the command in detached mode using tmux.

### Example Filters
Here are some example filters for common GPUs:

- **RTX 4090**:
  ```bash
  --filter "gpu_name=RTX_4090 num_gpus=4 reliability>=0.99"
  ```
- **H100 SXM**:
  ```bash
  --filter "gpu_name=H100_SXM num_gpus=1 reliability>=0.99"
  ```
- **A100 SXM4**:
  ```bash
  --filter "gpu_name=A100_SXM4 num_gpus=1 reliability>=0.9"
  ```

### Workflow
1. **Search and Select Offers**:
   - The script queries Vast.ai using the specified filter and displays the available GPU offers in a terminal interface.
   - If `--auto` is enabled, the first GPU is selected automatically.

2. **Create the Instance**:
   - The selected offer is used to create an instance, using the specified Docker image and environment variables.

3. **Copy Files**:
   - The script transfers your current directory to the remote GPU instance's workspace.

4. **Execute Command**:
   - The specified `--run` command is executed on the GPU instance.

5. **Clean Up**:
   - If `--keep` is not enabled, the instance is automatically destroyed after execution.

### Examples
- Automatically select a GPU with 100GB disk space and run a script:
  ```bash
  python auto_gpu.py --run "python train_reward_model.py" --disk 100 --auto
  ```

- Interactively select an H100 instance and keep it running after the script finishes:
  ```bash
  python auto_gpu.py --run "accelerate launch eval.py" --disk 150 --filter "gpu_name=H100_SXM" --keep
  ```

### Notes
- For the first instance, the script generates an `onstart.sh` file with commands to prepare the environment.
- If `--detach` is used, the script exits after starting the process in tmux on the remote instance.

### Troubleshooting
- If the script fails to parse offers or SSH fails, ensure your SSH key is added to Vast.ai and the filter matches available offers.
