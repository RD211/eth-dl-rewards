import os
import signal
import subprocess
import tempfile
from contextlib import contextmanager
from datasets import Dataset
from dataclasses import dataclass
from functools import partial
from latex2sympy2 import latex2sympy
from sympy import simplify
from typing import Optional

@dataclass
class Preference:
    accepted: str
    rejected: str

@dataclass
class Testcases:
    fn_name: Optional[str]
    input: list[str]
    output: list[str]

@dataclass
class GeneratedSolution:
    solution: str


def extract_code(response: GeneratedSolution):
    s = response.solution
    if '```' not in s:
        return s

    splitted = s.split('```')
    code = splitted[1]
    if code.startswith('python'):
        code = code[len('python'):]
        
    return code


def augment_code_with_testcases(code: str, testcases: Testcases) -> str:
    """
    If testcases.fn_name is provided, we append a loop that:
      - Calls that function with each input,
      - Compares the result to the expected output,
      - Increments a "passed" counter.
    Finally, it prints the number of testcases passed.

    Otherwise, we return a single code snippet that uses exec() to run 'code'
    for each test input, capturing the printed output, then counting how many
    match their expected output.
    """

    # Case 1: There's a function name => just append a test loop to 'code'.
    if testcases.fn_name:
        # We add typing to the function signature.
        code = f"from typing import List\n{code}"
        code = f"import heapq\nimport math\nimport sys\nimport collections\nfrom collections import *\nimport itertools\n{code}"
        # We'll store the input-output pairs in a variable, then loop over them.
        code += "\n\n"
        code += f"testcases_data = {list(zip(testcases.input, testcases.output))}\n"
        code += "passed = 0\n"
        code += "for inp, expected in testcases_data:\n"
        code += f"    result = Solution().{testcases.fn_name}(*eval(str(inp)))\n"  # or unpack if needed
        code += "    if str(result) == str(expected):\n"
        code += "        passed += 1\n"
        # code += "    else:\n"
        # code += "        print('Expected:', expected, 'Got:', result, 'for input:', inp)\n"
        code += "print(passed)\n"
        return code
    def preprocess_code(code: str) -> str:
        # First escape all backslashes
        code = code.replace('\\', '\\\\')
        # Then escape any triple quotes
        code = code.replace('"""', '\\"""')
        return code
    
    code = preprocess_code(code)
    # Case 2: No function name => wrap the entire code into a single runnable snippet.
    # We escape the code such that it works as a string literal.
    final_code = f"""
import io
import sys

solution_code = r\"\"\"{code}\"\"\"

# List of (input_string, output_string) pairs
testcases = {list(zip(testcases.input, testcases.output))}

def run_solution_with_input(code_str, input_data):
    \"\"\"
    Executes 'code_str' with the given string input_data.
    Captures printed output and returns it as a single string.
    \"\"\"
    original_stdin = sys.stdin
    original_stdout = sys.stdout
    try:
        # if input_data is a list, we join it with newlines
        if isinstance(input_data, list):
            input_data = '\\n'.join(input_data)
        sys.stdin = io.StringIO(input_data + "\\n")
        buffer = io.StringIO()
        sys.stdout = buffer
        exec(code_str, {{}})  # run in empty global/local
        return buffer.getvalue().strip()
    finally:
        sys.stdin = original_stdin
        sys.stdout = original_stdout

passed = 0
for inp, expected in testcases:
    actual_output = run_solution_with_input(solution_code, inp)
    # We strip newlines and compare the output
    actual_output = actual_output.strip()
    expected = expected.strip()
    # Remove all newlines at the end.
    expected = ','.join(expected.rstrip('\\n').split())
    actual_output = ','.join(actual_output.rstrip('\\n').split())
    if str(actual_output) == str(expected):
        passed += 1
    else:
        print('Expected:', expected, 'Got:', actual_output, "for input:", inp)

# Print number of passing tests
print(passed)
"""
    return final_code


def execute_codes(codes: list[str]) -> list[int]:
    ds = Dataset.from_list([{"code": code} for code in codes])
    postprocess_with_config = partial(postprocess_completion, code_config=code_config)
    result = ds.map(postprocess_with_config, num_proc=8)
    successes = []
    for _, row in enumerate(result):
        output_final = row["output"].split("\n")[-1]
        try:
            res = int(output_final)
        except:
            res = 0
            print("Error in output", row['output'], "for code", row["code"])
        successes.append(res)
    return successes

def code_preference(responses: list[GeneratedSolution], testcases: Testcases, max_number_of_pairs: int) -> list[Preference]:
    codes = [augment_code_with_testcases(extract_code(r), testcases) for r in responses]

    scores = execute_codes(codes)
    print("Scores", scores)
    # We generate preference pairs based on scores
    preference_pairs = []
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if scores[i] == scores[j]:
                continue
            if scores[i] > scores[j]:
                preference_pairs.append(Preference(accepted=responses[i].solution, rejected=responses[j].solution))
            else:
                    preference_pairs.append(Preference(accepted=responses[j].solution, rejected=responses[i].solution))


    return preference_pairs[:max_number_of_pairs]


# Define the CodeConfig class
class CodeConfig:
    def __init__(self, timeout):
        self.timeout = timeout  # Timeout for code execution

# Create a CodeConfig instance
code_config = CodeConfig(timeout=10)

class PythonREPL:
    def __init__(self, code_config):
        self.timeout = code_config.timeout

    @contextmanager
    def time_limit(self, seconds):
        def signal_handler(*_):
            raise TimeoutError(f"Timed out after {seconds} seconds.")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

    def __call__(self, queries):
        # We now need to combine all of them with the code template.
        query = queries[0]
        

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "tmp.py")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(query)
            with self.time_limit(self.timeout):
                result = subprocess.run(
                    ["python3", temp_file_path],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=self.timeout,
                )
                if result.returncode == 0:
                    return True, result.stdout.strip()
                error_msg = result.stderr.strip()
                msgs = error_msg.split("\n")
                new_msgs = []
                want_next = False
                for m in msgs:
                    if "Traceback" in m:
                        new_msgs.append(m)
                    elif m == msgs[-1]:
                        new_msgs.append(m)
                    elif temp_file_path in m:
                        st = m.index('"/') + 1 if '"/' in m else 0
                        ed = m.index(temp_file_path) + 1 if temp_file_path in m else None
                        clr = m[st:ed] if not ed else m[st:]
                        m = m.replace(clr, "")
                        new_msgs.append(m)
                        want_next = True
                    elif want_next:
                        new_msgs.append(m)
                        want_next = False
                error_msg = "\n".join(new_msgs)
                return False, error_msg.strip()
            
def execute_completion(executor, code):
    output = None    
    # We check for forbidden libraries.
    for lib in ("subprocess", "venv"):
        if lib in code:
            output = f"{lib} is not allowed"
            return output, False
    
    # We execute the code.
    try:
        success, output = executor([code])
        return output.strip(), success
    except TimeoutError as e:
        print("Code timed out")
        output = str(e)
        return output, False

def postprocess_completion(text, code_config):
    code_text = text["code"]
    executor = PythonREPL(code_config)
    output, success = execute_completion(executor, code_text)
    del executor
    return {
        "code": code_text,  # Preserve the original code
        "output": output,   # Add the output of the execution
        "success": success, # Add whether execution was successful
    }