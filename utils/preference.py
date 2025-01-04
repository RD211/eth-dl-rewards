import re
from dataclasses import dataclass
from latex2sympy2 import latex2sympy
from sympy import simplify
from .code_pairs import code_preference, Testcases, GeneratedSolution
@dataclass
class Preference:
  accepted: str
  rejected: str


def boxed_math_score(response, answer):
    def extract_or_default(text):
        # Try to extract content within \boxed{...}, fallback to the full text if not found
        match = re.search(r"\\boxed{(.*?)}", text)
        return match.group(1).strip() if match else text.strip()

    def try_parse_to_sympy(text):
        try:
            return latex2sympy(text)
        except Exception:
            return None

    # Extract the core content or use the original text
    response_core = extract_or_default(response)
    answer_core = extract_or_default(answer)

    # First, check if the extracted content matches directly
    if response_core == answer_core:
        return True

    # Attempt to parse both with latex2sympy and check symbolic equivalence
    response_sympy = try_parse_to_sympy(response_core)
    answer_sympy = try_parse_to_sympy(answer_core)

    if response_sympy is not None and answer_sympy is not None:
        try:
          return simplify(response_sympy - answer_sympy) == 0
        except Exception as e:
          print("There was an error in the simplification of the following expressions:")
          print(response_sympy)
          print(answer_sympy)
          print(e)
          return False

    # Fallback: they don't match and couldn't be parsed
    return False


def boxed_math_preference(responses: list[str], answer: str, max_number_of_pairs: int) -> list[Preference]:
  # We get scores for each response
  scores = [boxed_math_score(response, answer) for response in responses]
  
  # We generate preference pairs based on scores
  preference_pairs = []
  for i in range(len(scores)):
    for j in range(i + 1, len(scores)):
      if scores[i] == scores[j]:
        continue
      if scores[i]:
        preference_pairs.append(Preference(accepted=responses[i], rejected=responses[j]))
      else:
        preference_pairs.append(Preference(accepted=responses[j], rejected=responses[i]))


  return preference_pairs[:max_number_of_pairs]

def boxed_medical_score(response, answer, only_perfect_match=False):

  # We either have a single letter/string or a list of multiple strings
  def extract_answer(answer):
    if '[' in answer:
      try:
        return eval(answer)
      except:
        if ',' in answer:
          return answer.split(',')
        return answer
    if ',' in answer:
      return answer.split(',')
    return answer
  
  def extract_boxed(text):
    match = re.search(r"\\boxed{(.*?)}", text)
    return match.group(1).strip() if match else text.strip()
   
  generated_answer = extract_answer(extract_boxed(response))
  correct_answer = extract_answer(answer)
   
  # If they are different types, they are not equal
  if type(generated_answer) != type(correct_answer):
    return -1
  
  # If they are both strings, we compare them directly. We give -1 if they are not equal
  if type(generated_answer) == str:
    return 1 if generated_answer == correct_answer else -1
  
  # If they are both lists, we compare them element-wise. 
  # For each extra or missing element we subtract 1 out of the max score (length of the correct answer)
  # If all elements are correct, we return len(correct_answer)
  score = len(correct_answer)
  for i in range(len(correct_answer)):
    if correct_answer[i] not in generated_answer:
      score -= 1
  for i in range(len(generated_answer)):
    if generated_answer[i] not in correct_answer:
      score -= 1
  return score if not only_perfect_match else score == len(correct_answer)

def boxed_medical_preference(responses: list[str], answer: str, max_number_of_pairs: int) -> list[Preference]:
  # We get scores for each response
  scores = [boxed_medical_score(response, answer) for response in responses]
  
  # We generate preference pairs based on scores
  preference_pairs = []
  for i in range(len(scores)):
    for j in range(i + 1, len(scores)):
      if scores[i] == scores[j]:
        continue
      if scores[i] > scores[j]:
        preference_pairs.append(Preference(accepted=responses[i], rejected=responses[j]))
      else:
        preference_pairs.append(Preference(accepted=responses[j], rejected=responses[i]))


  return preference_pairs[:max_number_of_pairs]
  

def code_pref_wrapper(responses: list[str], testcases: str, max_number_of_pairs: int) -> list[Preference]:
  responses = [GeneratedSolution(solution=r) for r in responses]
  true = "true"
  false = "false"
  null = "null"
  Infinity = "Infinity"
  testcases = eval(testcases)
  if 'fn_name' in testcases:
      testcases = Testcases(
          fn_name=testcases['fn_name'],
          input=testcases['inputs'],
          output=testcases['outputs']
      )
  else:
      testcases = Testcases(
          fn_name=None,
          input=testcases['inputs'],
          output=testcases['outputs']
      )
  return code_preference(responses, testcases, max_number_of_pairs)

preference_function = {
  'math': boxed_math_preference,
  'medical': boxed_medical_preference,
  'code': code_pref_wrapper
}
