#!/usr/bin/env python3
"""
FSCS Adaptive Random Testing example.

Run from the command line, for example:
    python fscs_art_demo.py --tests 30 --candidates 10 --seed 0

Arguments:
    tests: number of test cases to run
    candidates: size of the candidate set C for FSCS
    seed: random seed for reproducibility

Domain: (weight_kg, distance_km)
  - weight_kg in [0, 50]
  - distance_km in [0, 500]

Distance metric:
  - Euclidean distance in a NORMALISED 2D space so both dimensions contribute fairly.

System under test:
  - A simple shipping-cost calculator with an injected bug that produces
    negative prices in a clustered region of the input space (heavy but
    short-distance shipments).
"""

import argparse
import math
import random
from typing import List, Tuple, Optional

TestInput = Tuple[float, float]  # (weight_kg, distance_km)

# Domain bounds (min, max) for each dimension
WEIGHT_RANGE = (0.0, 50.0)
DISTANCE_RANGE = (0.0, 500.0)


# ---------------------------
# Helpers for domain & distances
# ---------------------------

def random_input() -> TestInput:
    """Uniform random input (weight_kg, distance_km) in the domain."""
    w = random.uniform(*WEIGHT_RANGE)
    d = random.uniform(*DISTANCE_RANGE)
    return w, d


def _normalise(point: TestInput) -> Tuple[float, float]:
    """Map domain point into [0, 1] x [0, 1] for distance computation."""
    w, d = point
    w_min, w_max = WEIGHT_RANGE
    d_min, d_max = DISTANCE_RANGE
    w_norm = (w - w_min) / (w_max - w_min)
    d_norm = (d - d_min) / (d_max - d_min)
    return w_norm, d_norm


def euclidean_distance(p: TestInput, q: TestInput) -> float:
    """Euclidean distance between two inputs in normalised space."""
    pn = _normalise(p)
    qn = _normalise(q)
    return math.dist(pn, qn)  # Python 3.8+


def distance_to_nearest(x: TestInput, executed: List[TestInput]) -> float:
    """Distance from x to its nearest neighbour in the executed set."""
    return min(euclidean_distance(x, e) for e in executed)


# ---------------------------
# System under test
# ---------------------------

def shipping_cost(weight_kg: float, distance_km: float) -> float:
    """
    Example: shipping price calculator.

    Returns:
        A non-negative price in normal situations.

    BUG:
        For very heavy but short-distance shipments, a discount rule
        is implemented incorrectly and can make the price negative.
        This creates a contiguous "fault region" in the input space.

    Intended rule (for marketing):
        - Give a discount for heavy *long-distance* shipments:
            if weight > 30 kg and distance > 300 km: 10% off.

    Buggy implementation:
        - The comparison for distance is accidentally reversed
          (distance < 50 instead of distance > 300), so heavy but
          short-distance shipments get a huge discount.
    """
    if weight_kg < 0 or distance_km < 0:
        raise ValueError("Inputs must be non-negative")

    # Base cost: flat fee + per-kg + per-km
    base = 5.0 + 0.2 * weight_kg + 0.03 * distance_km

    # Correct discount: heavy & long distance → 10% off
    if weight_kg > 30 and distance_km > 300:
        base *= 0.9

    # BUG: wrong condition: heavy & *short* distance gets a big discount
    if weight_kg > 30 and distance_km < 50:
        base -= 40.0  # too large discount → may go negative

    return base


def system_under_test(test_input: TestInput) -> bool:
    """
    Execute the SUT and return True iff a failure is observed.

    Here we define a "failure" as the shipping cost becoming negative,
    which would be a clear symptom of the bug.
    """
    w, d = test_input
    price = shipping_cost(w, d)
    return price < 0.0


# ---------------------------
# FSCS-ART core algorithm
# ---------------------------

def select_next_fscs_test(executed: List[TestInput],
                          candidate_size: int) -> Tuple[TestInput, float]:
    """
    FSCS selection step:

      1. Generate 'candidate_size' random candidates in the input domain.
      2. For each candidate c, compute its distance to the nearest executed test.
      3. Return the candidate with the largest such distance (and that distance).
    """
    if not executed:
        raise ValueError("Executed set E must not be empty")

    best: Optional[TestInput] = None
    best_dist: float = -1.0

    for _ in range(candidate_size):
        c = random_input()
        d = distance_to_nearest(c, executed)
        if d > best_dist:
            best = c
            best_dist = d

    # 'best' must be set because candidate_size >= 1
    return best, best_dist  # type: ignore[return-value]


# ---------------------------
# Running a test campaign
# ---------------------------

def print_test_result(index: int,
                      x: TestInput,
                      min_dist: float,
                      failed: bool) -> None:
    """Pretty-print a single test result."""
    weight, distance = x
    status = "FAIL" if failed else "PASS"
    # For index == 1, min_dist is conventionally 0 (no previous tests).
    print(
        f"{index:3d}: weight={weight:5.2f} kg, "
        f"distance={distance:6.1f} km, "
        f"nearest_distance={min_dist:.3f}, "
        f"result={status}"
    )


def run_fscs_art(num_tests: int,
                 candidate_size: int,
                 seed: Optional[int] = None) -> None:
    """
    Run FSCS-ART on the shipping-cost SUT.

    - First test: plain random input.
    - Subsequent tests: FSCS selection (fixed-size candidate set).
    - Stop early if a failure is found.
    """
    if seed is not None:
        random.seed(seed)

    if num_tests <= 0:
        print("Nothing to do: num_tests must be > 0.")
        return

    executed: List[TestInput] = []
    failure_index: Optional[int] = None

    print(f"Running FSCS-ART on shipping_cost() "
          f"with num_tests={num_tests}, "
          f"candidate_size={candidate_size}, seed={seed}")
    print("-" * 80)

    # 1. Initial test: uniform random
    first = random_input()
    executed.append(first)
    failed = system_under_test(first)
    print_test_result(1, first, min_dist=0.0, failed=failed)

    if failed:
        failure_index = 1

    # 2. Remaining tests with FSCS
    for i in range(2, num_tests + 1):
        if failure_index is not None:
            break

        candidate, min_dist = select_next_fscs_test(executed, candidate_size)
        failed = system_under_test(candidate)
        executed.append(candidate)
        print_test_result(i, candidate, min_dist=min_dist, failed=failed)

        if failed:
            failure_index = i

    print("-" * 80)
    if failure_index is not None:
        print(f"Failure found after {failure_index} test(s).")
    else:
        print(f"No failure found after {len(executed)} test(s).")


# ---------------------------
# Command-line interface
# ---------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FSCS Adaptive Random Testing example on a shipping-cost calculator."
    )
    parser.add_argument(
        "--tests",
        type=int,
        default=40,
        help="Maximum number of test cases to run (default: 40).",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=10,
        help="Size of the candidate set C for FSCS (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )

    args = parser.parse_args()
    run_fscs_art(
        num_tests=args.tests,
        candidate_size=args.candidates,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
