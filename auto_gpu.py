import sys
import subprocess
import argparse
import time
import json
import curses
import os
import re
import shlex
from rich.console import Console

console = Console()

def search_vastai_offers(filter_string: str):
    command = f'vastai search offers "{filter_string}"'
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        if result.returncode != 0:
            console.print(f"[red]Error running vastai command:[/red] {result.stderr}")
            sys.exit(1)
        return result.stdout
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Command failed:[/red] {e.stderr}")
        sys.exit(1)

def parse_offers(offers_output):
    lines = offers_output.strip().splitlines()
    if not lines:
        console.print("[red]No offers found with the specified filter.[/red]")
        sys.exit(1)
    headers = list(filter(lambda x: x != 'Driver', lines[0].split()))
    rows = [line.split() for line in lines[1:]]
    return headers, rows

def calculate_column_widths(headers, rows):
    num_columns = len(headers)
    col_widths = [len(header) for header in headers]
    for row in rows:
        for i in range(num_columns):
            if i < len(row):
                col_widths[i] = max(col_widths[i], len(row[i]))
    return col_widths

def display_offers_curses(stdscr, headers, rows):
    curses.curs_set(0)
    index = 0
    col_widths = calculate_column_widths(headers, rows)
    height, width = stdscr.getmaxyx()
    max_display_rows = height - 5
    start_row = 0

    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Available GPU Offers")
        header_items = [header.ljust(col_widths[i]) for i, header in enumerate(headers)]
        stdscr.addstr(1, 0, "  ".join(header_items)[:width-1], curses.A_BOLD)
        end_row = start_row + max_display_rows
        display_rows = rows[start_row:end_row]

        for i, row in enumerate(display_rows):
            row_index = start_row + i
            row_items = [row[j].ljust(col_widths[j]) for j in range(len(row))]
            row_str = "  ".join(row_items)
            if row_index == index:
                stdscr.addstr(i + 2, 0, row_str[:width-1], curses.A_REVERSE)
            else:
                stdscr.addstr(i + 2, 0, row_str[:width-1])

        stdscr.addstr(height - 2, 0, "Use ↑ and ↓ to navigate, Enter to select.", curses.A_DIM)
        stdscr.refresh()
        key = stdscr.getch()
        if key == curses.KEY_UP:
            if index > 0:
                index -= 1
                if index < start_row:
                    start_row -= 1
        elif key == curses.KEY_DOWN:
            if index < len(rows) - 1:
                index += 1
                if index >= start_row + max_display_rows:
                    start_row += 1
        elif key == ord('\n'):
            return rows[index][0]

def interactive_selection(headers, rows):
    return curses.wrapper(display_offers_curses, headers, rows)

def write_onstart_script(onstart_cmd: str, auto_gpu_dir: str):
    onstart_script_path = os.path.join(auto_gpu_dir, "onstart.sh")

    with open(onstart_script_path, "w") as file:
        file.write("#!/bin/bash\n")
        file.write(onstart_cmd + "\n")

    os.chmod(onstart_script_path, 0o755)

    return "onstart.sh"

def parse_ssh_url(ssh_url: str):
    # ssh_url format: ssh://user@host:port
    pattern = r"ssh://(.*)@(.*):(\d+)"
    match = re.match(pattern, ssh_url)
    if match:
        user = match.group(1)
        host = match.group(2)
        port = match.group(3)
        return user, host, port
    else:
        console.print(f"[red]Invalid SSH URL format: {ssh_url}[/red]")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Auto GPU Selection Script')
    parser.add_argument('--filter', type=str, default='', help='Filter string for vast.ai search')
    parser.add_argument('--run', type=str, required=True, help='Command to run on the remote server')
    parser.add_argument('--disk', type=float, help='Required disk capacity in GB')
    parser.add_argument('--auto', action='store_true', help='Automatically select an instance without prompting')
    parser.add_argument('--image', type=str, default="pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime", help='Docker image to use on the instance')
    parser.add_argument('--env', type=str, default="-e DATA_DIRECTORY=/workspace/ -e JUPYTER_DIR=/", help="Environment variables for Docker container")
    parser.add_argument('--onstart', type=str, default="env >> /etc/environment; mkdir -p ${DATA_DIRECTORY:-/workspace/}; touch ~/.no_auto_tmux", help="Onstart command for Docker container")
    parser.add_argument('--keep', action='store_true', help='Keep the instance after the script finishes')
    parser.add_argument('--detach', action='store_true', help='Do not view the script output and exit immediately')

    args = parser.parse_args()

    filter_string = args.filter
    run_command = args.run
    disk_capacity = args.disk
    auto_select = args.auto
    image = args.image
    env_vars = args.env
    onstart_cmd = args.onstart
    keep_instance = args.keep
    detach = args.detach

    if not run_command:
        console.print("[red]Error: The --run parameter is required.[/red]")
        sys.exit(1)

    if disk_capacity is None:
        try:
            disk_input = console.input("Enter required disk capacity in GB: ")
            disk_capacity = float(disk_input)
        except ValueError:
            console.print("[red]Invalid disk capacity entered. Exiting.[/red]")
            sys.exit(1)

    if filter_string:
        filter_string = f"{filter_string} disk_space > {disk_capacity}"
    else:
        filter_string = f"disk_space > {disk_capacity}"

    # Step 0: Create .auto_gpu directory
    auto_gpu_dir = ".auto_gpu"
    os.makedirs(auto_gpu_dir, exist_ok=True)

    # Step 1: Search and display offers
    offers_output = search_vastai_offers(filter_string)
    headers, rows = parse_offers(offers_output)

    if auto_select:
        selected_offer_id = rows[0][0]
        console.print(f"[green]Automatically selected offer ID:[/green] {selected_offer_id}")
    else:
        selected_offer_id = interactive_selection(headers, rows)

    onstart_script_path = write_onstart_script(onstart_cmd, auto_gpu_dir)

    instance_id = None
    try:
        # Step 2: Create the instance inside .auto_gpu directory
        console.print(f"[cyan]Creating instance from offer ID {selected_offer_id}...[/cyan]")
        create_command = [
            'vastai', 'create', 'instance', selected_offer_id,
            '--image', image,
            '--disk', str(int(disk_capacity)),
            '--ssh', '--direct',
            '--raw',
            '--env', env_vars,
            '--onstart', onstart_script_path
        ]

        try:
            result = subprocess.run(create_command, check=True, text=True, capture_output=True, cwd=auto_gpu_dir)
            output = result.stdout.strip()

            output_json = json.loads(output)
            if output_json.get('success'):
                instance_id = str(output_json.get('new_contract'))
                console.print(f"[green]Instance created successfully with ID: {instance_id}[/green]")
            else:
                console.print(f"[red]Failed to create instance:[/red] {output_json}")
                sys.exit(1)
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error creating instance:[/red] {e.stderr}")
            sys.exit(1)
        except json.JSONDecodeError:
            console.print(f"[red]Failed to parse JSON output from 'vastai create instance':[/red] {output}")
            sys.exit(1)

        # Step 3: Wait for the instance to be running
        console.print("[cyan]Waiting for the instance to be running...[/cyan]")
        instance_running = False
        while not instance_running:
            try:
                result = subprocess.run(['vastai', 'show', 'instances', '--raw'], check=True, text=True, capture_output=True, cwd=auto_gpu_dir)
                output = result.stdout.strip()
                instances = json.loads(output)
                for instance in instances:
                    if str(instance['id']) == instance_id:
                        status = instance.get('actual_status', 'unknown')
                        console.print(f"Instance status: {status}")
                        if status == 'running':
                            console.print(f"[green]Instance {instance_id} is running.[/green]")
                            instance_running = True
                            time.sleep(30)  # Wait extra time for SSH to be ready
                        elif status == 'failed':
                            console.print(f"[red]Instance {instance_id} failed to start.[/red]")
                            sys.exit(1)
                        else:
                            time.sleep(10)  # Wait before checking again
                        break
                else:
                    console.print(f"[red]Instance {instance_id} not found in 'vastai show instances' output.[/red]")
                    sys.exit(1)
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Error checking instance status:[/red] {e.stderr}")
                sys.exit(1)
            except json.JSONDecodeError:
                console.print(f"[red]Failed to parse JSON output from 'vastai show instances':[/red] {output}")
                sys.exit(1)

        # Step 4: Copy current directory to instance (run from the original directory)
        console.print("[cyan]Copying current directory to instance...[/cyan]")
        copy_command = ['vastai', 'copy', './', f'{instance_id}:/workspace']
        try:
            result = subprocess.run(copy_command, check=True, text=True)
            console.print("[green]Files copied successfully.[/green]")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error copying files to instance:[/red] {e.stderr}")
            sys.exit(1)

        # Step 5: Get the SSH URL (run inside .auto_gpu)
        console.print("[cyan]Retrieving SSH URL...[/cyan]")
        ssh_url_command = ['vastai', 'ssh-url', instance_id]
        try:
            result = subprocess.run(ssh_url_command, check=True, text=True, capture_output=True, cwd=auto_gpu_dir)
            ssh_url = result.stdout.strip()
            console.print(f"[green]SSH URL retrieved: {ssh_url}[/green]")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error getting SSH URL:[/red] {e.stderr}")
            sys.exit(1)

        # Step 6: Parse SSH URL
        user, host, port = parse_ssh_url(ssh_url)

        # Step 7: Prepare to execute the run command on the instance via SSH

        # Get the VAST.AI API key from the local machine
        api_key_path = os.path.expanduser("~/.vast_api_key")
        if not os.path.exists(api_key_path):
            console.print("[red]VAST.AI API key file not found. Exiting.[/red]")
            sys.exit(1)

        with open(api_key_path, "r") as file:
            vast_ai_key = file.read().strip()

        # Escape single quotes in variables to prevent breaking the shell command
        def escape_single_quotes(s):
            return s.replace("'", "'\\''")

        vast_ai_key_escaped = escape_single_quotes(vast_ai_key)
        run_command_escaped = escape_single_quotes(run_command)
        keep_instance_str = "True" if keep_instance else "False"

        # Build the remote commands
        remote_commands = f"""
        set +e ; \
        cd /workspace ; \
        ls -la /workspace ; \
        chmod +x setup.sh ; \
        ./setup.sh ; \
        pip install vastai ; \
        echo '{vast_ai_key_escaped}' > ~/.vast_api_key ; \
        {run_command_escaped}; \
        {"" if keep_instance_str == 'True' else f"vastai destroy instance {instance_id}; sleep 10;"}
        """


        # Remove leading/trailing whitespace and newlines
        remote_commands = remote_commands.strip().replace('\n', ' ')

        # Escape the remote commands
        escaped_remote_commands = shlex.quote(remote_commands)

        # Build the SSH command
        ssh_command_str = f'ssh -t -o StrictHostKeyChecking=no -p {port} {user}@{host} tmux new {"-A" if not detach else "-d"} -s my_session "{escaped_remote_commands}"'

        # Step 8: Execute the run command on the instance via SSH (run in /workspace)
        console.print("[cyan]Executing command on the instance via SSH...[/cyan]")
        try:
            result = subprocess.run(ssh_command_str, shell=True)
            console.print("[green]Command executed successfully on the instance.[/green]")
            if not keep_instance and not detach:
                try:
                    console.print(f"[cyan]Destroying instance {instance_id}...[/cyan]")
                    destroy_command = ['vastai', 'destroy', 'instance', instance_id]
                    try:
                        result = subprocess.run(destroy_command, check=True, text=True, cwd=auto_gpu_dir)
                        console.print(f"[green]Instance {instance_id} destroyed successfully.[/green]")
                    except subprocess.CalledProcessError as e:
                        console.print(f"[red]Error destroying instance:[/red] {e.stderr}")
                except:
                    console.print(f"[red]Error destroying instance {instance_id}. Please destroy manually if not already dead.[/red]")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error executing command on instance via SSH:[/red] {e.stderr}")
            sys.exit(1)
    except:
        if instance_id:
            console.print(f"[cyan]Destroying instance {instance_id}...[/cyan]")
            destroy_command = ['vastai', 'destroy', 'instance', instance_id]
            try:
                result = subprocess.run(destroy_command, check=True, text=True, cwd=auto_gpu_dir)
                console.print(f"[green]Instance {instance_id} destroyed successfully.[/green]")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Error destroying instance:[/red] {e.stderr}")

if __name__ == "__main__":
    main()