import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz.distance import Levenshtein

load_dotenv()

PROMPT_TEMPLATE = """
You are triaging multi-service incidents. Each log snippet begins with a subsystem tag such as [AUTH] or [BILL].
Select the telemetry tool that matches the subsystem tag, call it exactly once, and use the returned metrics to summarize the service health.
Do not attempt to answer without making a tool call. The only reliable metrics are inside the tool response JSON.

Available tools:
{tool_guidance}

Log snippet: {input_value}

After receiving the tool output, reply with:

Result: <SERVICE_TAG> error_rate=<PERCENT>% p99_latency_ms=<NUMBER>
"""

SYSTEM_PROMPT = "You follow instructions exactly and always call the matching subsystem tool before answering."

MODEL_NAME = os.getenv("LLM_MODEL", "gpt-4o-mini")
CLIENT = OpenAI()

ToolEvent = Tuple[str, bool]

@dataclass(frozen=True)
class ToolVariant:
    name: str
    prefix: str
    description: str


TOOL_VARIANTS: Sequence[ToolVariant] = (
    ToolVariant(
        name="pull_auth_telemetry",
        prefix="[AUTH]",
        description="Fetches live authentication telemetry for [AUTH] logs, returning JSON with error_rate and p99_latency_ms.",
    ),
    ToolVariant(
        name="pull_billing_telemetry",
        prefix="[BILL]",
        description="Fetches live billing telemetry for [BILL] logs, returning JSON with error_rate and p99_latency_ms.",
    ),
    ToolVariant(
        name="pull_fulfillment_telemetry",
        prefix="[FULF]",
        description="Fetches live fulfillment telemetry for [FULF] logs, returning JSON with error_rate and p99_latency_ms.",
    ),
    ToolVariant(
        name="pull_risk_telemetry",
        prefix="[RISK]",
        description="Fetches live risk-engine telemetry for [RISK] logs, returning JSON with error_rate and p99_latency_ms.",
    ),
    ToolVariant(
        name="pull_support_telemetry",
        prefix="[SUPP]",
        description="Fetches live support-hub telemetry for [SUPP] logs, returning JSON with error_rate and p99_latency_ms.",
    ),
)

TOOL_DEFINITIONS = [
    {
        "name": variant.name,
        "type": "function",
        "function": {
            "name": variant.name,
            "description": variant.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "log_excerpt": {
                        "type": "string",
                        "description": "Log snippet text whose telemetry metrics should be estimated.",
                    }
                },
                "required": ["log_excerpt"],
            },
        },
    }
    for variant in TOOL_VARIANTS
]

TOOL_PREFIX_LOOKUP: Dict[str, str] = {variant.name: variant.prefix for variant in TOOL_VARIANTS}
TOOL_PREFIXES: Tuple[str, ...] = tuple(variant.prefix for variant in TOOL_VARIANTS)

LOG_BODY_PHRASES: Tuple[str, ...] = (
    "error spike after {ms}ms handshake in {region}",
    "retry queue length {count}k and backlog warning in {region}",
    "timeout alerts for {count} shards, fail open triggered in {region}",
    "denied tokens causing {count} consecutive fail events in {region}",
    "latency holding at {ms}ms p99, rerouting traffic in {region}",
    "partial outage noted; {count} retries before success in {region}",
    "backlog draining slowly, queue depth {count}k in {region}",
)

TOOL_GUIDANCE = "\n".join(
    f"- {variant.name}: {variant.description}" for variant in TOOL_VARIANTS
)


def tool_matches_input(tool_name: str, text_input: str) -> bool:
    prefix = TOOL_PREFIX_LOOKUP.get(tool_name)
    if not prefix:
        return False
    return text_input.startswith(prefix)


def summarize_log(text_input: str) -> Dict[str, float]:
    lowered = text_input.lower()
    error_tokens = sum(lowered.count(keyword) for keyword in ("error", "fail", "denied"))
    timeout_tokens = lowered.count("timeout") + lowered.count("latency")
    retry_tokens = sum(lowered.count(keyword) for keyword in ("retry", "queue", "backlog"))
    total_tokens = max(1, len(lowered.split()))
    error_rate = min(100.0, 3.0 + error_tokens * 8.5 + timeout_tokens * 2.5)
    latency = 45 + (len(text_input) % 70) + retry_tokens * 5 + timeout_tokens * 7
    backlog = retry_tokens + max(0, error_tokens - 1)
    return {
        "error_rate": round(error_rate, 1),
        "p99_latency_ms": latency,
        "retry_backlog": backlog,
        "tokens_considered": total_tokens,
    }


def extract_tool_events_from_output(blocks: Optional[Sequence[object]], text_input: str) -> List[ToolEvent]:
    events: List[ToolEvent] = []
    if not blocks:
        return events
    for block in blocks:
        block_type = getattr(block, "type", None)
        if block_type != "function_call":
            continue
        tool_name = getattr(block, "name", "")
        events.append((tool_name, tool_matches_input(tool_name, text_input)))
    return events


def call_model(text_input: str) -> Tuple[str, List[ToolEvent]]:
    prompt = PROMPT_TEMPLATE.format(input_value=text_input, tool_guidance=TOOL_GUIDANCE)
    response = CLIENT.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tools=TOOL_DEFINITIONS,
    )
    tool_events: List[ToolEvent] = []

    print(f"Response: {response}")

    while response.status == "requires_action" and response.required_action:
        tool_calls = response.required_action.submit_tool_outputs.tool_calls
        tool_outputs = []
        for tool_call in tool_calls:
            arguments = json.loads(tool_call.function.arguments or "{}")
            provided_text = arguments.get("log_excerpt") or arguments.get("text") or text_input
            summary = summarize_log(provided_text)
            tool_outputs.append(
                {
                    "tool_call_id": tool_call.id,
                    "output": json.dumps(summary),
                }
            )
            tool_events.append(
                (
                    tool_call.function.name,
                    tool_matches_input(tool_call.function.name, provided_text),
                )
            )
        response = CLIENT.responses.submit_tool_outputs(
            response_id=response.id,
            tool_outputs=tool_outputs,
        )

    tool_events.extend(extract_tool_events_from_output(response.output, text_input))

    text = "".join(
        piece.text
        for block in response.output
        for piece in getattr(block, "content", [])
        if getattr(piece, "type", "") == "output_text"
    )

    return text, tool_events


def random_log_line(rng: random.Random, length_range: Tuple[int, int]) -> str:
    low, high = length_range
    target_words = rng.randint(low, high)
    segments: List[str] = []
    while len(" ".join(segments).split()) < target_words:
        template = rng.choice(LOG_BODY_PHRASES)
        segment = template.format(
            ms=rng.randint(40, 280),
            region=rng.choice(TOOL_PREFIXES),
            count=rng.randint(1, 12),
        )
        segments.append(segment)
    prefix = rng.choice(TOOL_PREFIXES)
    body = " | ".join(segments)
    return f"{prefix} {body}"


def select_fscs_candidate(
    previous: List[str],
    pool: int,
    rng: random.Random,
    length_range: Tuple[int, int],
) -> str:
    if not previous:
        return random_log_line(rng, length_range)
    best_candidate: Optional[str] = None
    best_distance = -1
    for _ in range(pool):
        candidate = random_log_line(rng, length_range)
        distance = min(Levenshtein.distance(candidate, seen) for seen in previous)
        if distance > best_distance:
            best_candidate = candidate
            best_distance = distance
    return best_candidate if best_candidate is not None else random_log_line(rng, length_range)


def adaptive_random_testing(
    *,
    pool_size: int = 10,
    max_iterations: int = 5,
    seed: Optional[int] = 1234,
    length_range: Tuple[int, int] = (12, 28),
) -> Tuple[List[Tuple[int, str, str, List[ToolEvent]]], int, int, int]:
    rng = random.Random(seed)
    tested: List[str] = []
    samples: List[Tuple[int, str, str, List[ToolEvent]]] = []
    correct_tool_calls = 0
    total_tool_calls = 0
    for iteration in range(1, max_iterations + 1):
        candidate = select_fscs_candidate(tested, pool_size, rng, length_range)
        raw, tool_events = call_model(candidate)
        tested.append(candidate)
        if len(samples) < 5:
            samples.append((iteration, candidate, raw, tool_events))
        for _, is_correct in tool_events:
            total_tool_calls += 1
            if is_correct:
                correct_tool_calls += 1
    return samples, correct_tool_calls, total_tool_calls


def main() -> None:
    samples, correct_tool_calls, total_tool_calls = adaptive_random_testing()
    if samples:
        print("Sample responses:")
        for iteration, value, raw, tool_events in samples:
            print(f"- Iteration {iteration}, input {value}")
            if tool_events:
                breakdown = ", ".join(
                    f"{name} ({'correct' if was_correct else 'incorrect'})"
                    for name, was_correct in tool_events
                )
                print(f"  Tool calls: {breakdown}")
            else:
                print("  Tool calls: none recorded.")
            print("  Raw response snippet:")
            print("  " + raw.strip().replace("\n", "\n  "))
    else:
        print("No model responses were recorded.")


    if total_tool_calls:
        accuracy = (correct_tool_calls / total_tool_calls) * 100
        print(
            f"Tool selection accuracy: {correct_tool_calls}/{total_tool_calls} ({accuracy:.1f}%)"
        )
    else:
        print("Tool selection accuracy: 0/0 (model never called a tool).")


if __name__ == "__main__":
    main()
