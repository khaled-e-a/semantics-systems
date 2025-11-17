import os
import random
import string
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz.distance import Levenshtein

load_dotenv()

PROMPT_TEMPLATE = """
Given the string below, calculate the number of occurrences of the letter 'r'.

Input: {input_value}

Your final result must be in the following format:

Result: NUMBER
"""

MODEL_NAME = os.getenv("LLM_MODEL", "gpt-4o-mini")
CLIENT = OpenAI()


def call_model(text_input: str) -> Tuple[int, str]:
    prompt = PROMPT_TEMPLATE.format(input_value=text_input)
    response = CLIENT.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": "You follow instructions exactly."},
            {"role": "user", "content": prompt},
        ],
    )
    text = "".join(
        piece.text
        for block in response.output
        for piece in getattr(block, "content", [])
        if getattr(piece, "type", "") == "output_text"
    )
    num_line = [line for line in text.splitlines() if line.startswith("Result: ")][-1]
    num = int(num_line.split(": ", 1)[-1])
    return num, text


def random_string(rng: random.Random, length_range: Tuple[int, int]) -> str:
    low, high = length_range
    length = rng.randint(low, high)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


def select_fscs_candidate(
    previous: List[str],
    pool: int,
    rng: random.Random,
    length_range: Tuple[int, int],
) -> str:
    if not previous:
        return random_string(rng, length_range)
    best_candidate: Optional[str] = None
    best_distance = -1
    for _ in range(pool):
        candidate = random_string(rng, length_range)
        distance = min(Levenshtein.distance(candidate, seen) for seen in previous)
        if distance > best_distance:
            best_candidate = candidate
            best_distance = distance
    return best_candidate if best_candidate is not None else random_string(rng, length_range)


def adaptive_random_testing(
    *,
    pool_size: int = 10,
    max_iterations: int = 100,
    seed: Optional[int] = 1234,
    length_range: Tuple[int, int] = (5, 12),
) -> Optional[Tuple[str, int, int, str, int]]:
    rng = random.Random(seed)
    tested: List[str] = []
    for iteration in range(1, max_iterations + 1):
        candidate = select_fscs_candidate(tested, pool_size, rng, length_range)
        actual, raw = call_model(candidate)
        expected = candidate.lower().count("r")
        tested.append(candidate)
        if actual != expected:
            return candidate, expected, actual, raw, iteration
    return None


def main() -> None:
    failure = adaptive_random_testing()
    if not failure:
        print("No failure discovered within the iteration budget.")
        return
    value, expected, actual, raw, iteration = failure
    print(f"Failure after {iteration} iterations on input {value}.")
    print(f"Expected: {expected}, actual: {actual}")
    print("Raw response:")
    print(raw.strip())


if __name__ == "__main__":
    main()
