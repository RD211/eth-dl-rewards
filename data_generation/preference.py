import re
from dataclasses import dataclass
from latex2sympy2 import latex2sympy
from sympy import simplify

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
  



preference_function = {
  'math': boxed_math_preference,
}
