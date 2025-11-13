#!/usr/bin/env python3
# mbfl_demo.py  — Mutation-Based Fault Localization

import ast
import copy
import textwrap
import types
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Any

# ---------- 1) Buggy program under test ----------
ORIGINAL_SOURCE = textwrap.dedent("""
def triangle_type(a, b, c):
    # Determine triangle type or INVALID.
    # BUG: uses '>=' for triangle inequality; should be '>'
    if a <= 0 or b <= 0 or c <= 0:
        return "INVALID"
    if a + b >= c and a + c >= b and b + c >= a:  # bug line
        if a == b == c:
            return "EQUILATERAL"
        if a == b or a == c or b == c:
            return "ISOSCELES"
        return "SCALENE"
    return "INVALID"
""").strip()


# ---------- 2) Tiny test suite (no pytest needed) ----------
Test = Callable[[Callable[..., str]], None]

def t_scalene_ok(f):
    assert f(3, 4, 5) == "SCALENE"

def t_isosceles_ok(f):
    assert f(2, 2, 3) == "ISOSCELES"

def t_equilateral_ok(f):
    assert f(5, 5, 5) == "EQUILATERAL"

def t_invalid_nonpositive(f):
    assert f(0, 2, 3) == "INVALID"

def t_invalid_triangle_inequality(f):
    # This should be INVALID; the buggy '>=' will misclassify it.
    assert f(2, 2, 4) == "INVALID"

TESTS: List[Tuple[str, Test]] = [
    ("t_scalene_ok", t_scalene_ok),
    ("t_isosceles_ok", t_isosceles_ok),
    ("t_equilateral_ok", t_equilateral_ok),
    ("t_invalid_nonpositive", t_invalid_nonpositive),
    ("t_invalid_triangle_inequality", t_invalid_triangle_inequality),
]


# ---------- 3) Mutant representation ----------
@dataclass
class Mutant:
    id: int
    lineno: int
    description: str
    mutated_source: str


# ---------- 4) Mutation engine (single-edit mutants, precise targeting) ----------
OP_SWAPS = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.Div: ast.Mult,
    ast.GtE: ast.Gt,
    ast.Gt: ast.GtE,
    ast.LtE: ast.Lt,
    ast.Lt: ast.LtE,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.And: ast.Or,
    ast.Or: ast.And,
    ast.USub: ast.UAdd,
    ast.UAdd: ast.USub,
}

def _swap_op(node):
    T = type(node)
    if T in OP_SWAPS:
        return OP_SWAPS[T]()
    return None


@dataclass
class Site:
    kind: str                     # "binop" | "boolop" | "unary" | "compare"
    lineno: int
    col_offset: int
    compare_op_index: Optional[int] = None  # for Compare nodes only


class SiteCollector(ast.NodeVisitor):
    """
    Collect candidate operator nodes precisely:
    - For BinOp/BoolOp/UnaryOp: store (kind, lineno, col_offset)
    - For Compare: store a separate site for EACH comparator op with its index
    """
    def __init__(self):
        self.sites: List[Site] = []

    def visit_BinOp(self, node: ast.BinOp):
        if _swap_op(node.op):
            self.sites.append(Site("binop", node.lineno, node.col_offset))
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp):
        if _swap_op(node.op):
            self.sites.append(Site("boolop", node.lineno, node.col_offset))
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if _swap_op(node.op):
            self.sites.append(Site("unary", node.lineno, node.col_offset))
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        for i, op in enumerate(node.ops):
            if _swap_op(op):
                self.sites.append(Site("compare", node.lineno, node.col_offset, compare_op_index=i))
        self.generic_visit(node)


def create_mutants(source: str) -> List[Mutant]:
    tree = ast.parse(source)
    collector = SiteCollector()
    collector.visit(tree)

    mutants: List[Mutant] = []
    mid = 1

    for site in collector.sites:
        m_tree = copy.deepcopy(tree)

        class SingleEdit(ast.NodeTransformer):
            edited = False

            # Match on both lineno and col_offset (plus op index for Compare)
            def visit_BinOp(self, n: ast.BinOp):
                if (not self.edited and
                    site.kind == "binop" and
                    getattr(n, "lineno", -1) == site.lineno and
                    getattr(n, "col_offset", -1) == site.col_offset):
                    newop = _swap_op(n.op)
                    if newop:
                        self.edited = True
                        return ast.copy_location(ast.BinOp(left=n.left, op=newop, right=n.right), n)
                return self.generic_visit(n)

            def visit_BoolOp(self, n: ast.BoolOp):
                if (not self.edited and
                    site.kind == "boolop" and
                    getattr(n, "lineno", -1) == site.lineno and
                    getattr(n, "col_offset", -1) == site.col_offset):
                    newop = _swap_op(n.op)
                    if newop:
                        self.edited = True
                        return ast.copy_location(ast.BoolOp(op=newop, values=n.values), n)
                return self.generic_visit(n)

            def visit_UnaryOp(self, n: ast.UnaryOp):
                if (not self.edited and
                    site.kind == "unary" and
                    getattr(n, "lineno", -1) == site.lineno and
                    getattr(n, "col_offset", -1) == site.col_offset):
                    newop = _swap_op(n.op)
                    if newop:
                        self.edited = True
                        return ast.copy_location(ast.UnaryOp(op=newop, operand=n.operand), n)
                return self.generic_visit(n)

            def visit_Compare(self, n: ast.Compare):
                if (not self.edited and
                    site.kind == "compare" and
                    getattr(n, "lineno", -1) == site.lineno and
                    getattr(n, "col_offset", -1) == site.col_offset and
                    site.compare_op_index is not None and
                    0 <= site.compare_op_index < len(n.ops)):
                    new_ops = list(n.ops)
                    i = site.compare_op_index
                    swapped = _swap_op(new_ops[i])
                    if swapped:
                        new_ops[i] = swapped
                        self.edited = True
                        return ast.copy_location(
                            ast.Compare(left=n.left, ops=new_ops, comparators=n.comparators),
                            n
                        )
                return self.generic_visit(n)

        transformer = SingleEdit()
        new_tree = transformer.visit(m_tree)
        if not transformer.edited:
            # No valid swap for this site (should be rare due to collection checks)
            continue

        ast.fix_missing_locations(new_tree)
        src_text = ast.unparse(new_tree)
        desc = f"{site.kind} swap at line {site.lineno}, col {site.col_offset}"
        if site.kind == "compare":
            desc += f", op_index {site.compare_op_index}"
        mutants.append(Mutant(id=mid, lineno=site.lineno, description=desc, mutated_source=src_text))
        mid += 1

    return mutants


# ---------- 5) Test runner & kill matrix ----------
def _load_func_from_source(src_text: str) -> Callable[..., str]:
    mod = types.ModuleType("mutant_mod")
    exec(compile(src_text, "<mem>", "exec"), mod.__dict__)
    return getattr(mod, "triangle_type")

def _baseline_func() -> Callable[..., str]:
    mod = types.ModuleType("orig_mod")
    exec(compile(ORIGINAL_SOURCE, "<orig>", "exec"), mod.__dict__)
    return getattr(mod, "triangle_type")

def run_suite(f) -> Dict[str, bool]:
    results = {}
    for name, test in TESTS:
        try:
            test(f)
            results[name] = True
        except AssertionError:
            results[name] = False
    return results

def kill_vector(mutant_src_text: str, baseline_outcomes: Dict[str, bool]) -> Dict[str, bool]:
    f = _load_func_from_source(mutant_src_text)
    outcomes = run_suite(f)
    # A test "kills" the mutant if its pass/fail differs from baseline
    return {tname: (outcomes[tname] != baseline_outcomes[tname]) for tname in baseline_outcomes}


# ---------- 6) Ochiai per mutant & line aggregation ----------
def ochiai_score(kills: Dict[str, bool], baseline_outcomes: Dict[str, bool]) -> float:
    F = sum(not ok for ok in baseline_outcomes.values())
    if F == 0:
        return 0.0
    Kf = sum(kills[t] for t, ok in baseline_outcomes.items() if not ok)
    Kp = sum(kills[t] for t, ok in baseline_outcomes.items() if ok)
    denom = (F * (Kf + Kp)) ** 0.5
    return (Kf / denom) if denom else 0.0

def rank_lines(mutants: List[Mutant], baseline_outcomes: Dict[str, bool]) -> List[Tuple[int, float, List[int]]]:
    # returns list of (lineno, score, mutant_ids_contributing)
    line_to_scores: Dict[int, List[Tuple[int, float]]] = {}
    for m in mutants:
        kills = kill_vector(m.mutated_source, baseline_outcomes)
        score = ochiai_score(kills, baseline_outcomes)
        line_to_scores.setdefault(m.lineno, []).append((m.id, score))
    ranking = []
    for lineno, pairs in line_to_scores.items():
        max_score = max(s for _, s in pairs) if pairs else 0.0
        contrib = [mid for mid, s in pairs if abs(s - max_score) < 1e-12]
        ranking.append((lineno, max_score, contrib))
    ranking.sort(key=lambda x: x[1], reverse=True)
    return ranking


# ---------- 7) End-to-end: run MBFL ----------
def main():
    print("Original program:\n", ORIGINAL_SOURCE, "\n", sep="")
    baseline_f = _baseline_func()
    baseline_outcomes = run_suite(baseline_f)
    print("Baseline test results:")
    for t, ok in baseline_outcomes.items():
        print(f"  {t}: {'PASS' if ok else 'FAIL'}")
    F = sum(not ok for ok in baseline_outcomes.values())
    if F == 0:
        print("\nNo failing tests; MBFL needs at least one failing test. Exiting.")
        return

    mutants = create_mutants(ORIGINAL_SOURCE)
    print(f"\nGenerated {len(mutants)} single-edit mutants.")
    for m in mutants[:8]:
        print(f"  Mutant {m.id} @ line {m.lineno}: {m.description}")
    if len(mutants) > 8:
        print("  ...")

    ranking = rank_lines(mutants, baseline_outcomes)
    print("\nTop suspicious lines (Ochiai, max per line):")
    for lineno, score, contrib in ranking[:10]:
        print(f"  line {lineno:>3}  score={score:.4f}  via mutants {contrib}")

if __name__ == "__main__":
    main()
