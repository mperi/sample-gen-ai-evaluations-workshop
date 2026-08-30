"""Headless live run of the expected-behavior conformance suite.

CLI equivalent of the notebook's "Step 8" — for running the full evaluation
without opening Jupyter (CI, quick re-runs, etc.).

Mirrors the sequential notebook's run_peers/run_session logic (the three peers
share a memory list, source_facts threaded per turn) and grades all briefs in
datasets/research_briefs.jsonl against their expected_behavior via the LLM judge.

NOTE: this intentionally duplicates the notebook's pipeline logic because
notebook cells are not importable. If you change the pipeline in
04-peer-to-peer-sequential.ipynb, mirror the change here (and vice versa).

Makes real Bedrock calls. Requires AWS credentials and Bedrock model access in
the target region (AWS_REGION, default us-west-2).

Run:  python run_behavior_live.py
"""
import os
import time
import logging

from strands import Agent

from model_config import AGENT_MODEL_ID, PEER_CONFIGS
from eval_helpers import format_memory
from metrics_collector import MetricsCollector, LLMJudge
from behavior_eval import run_behavior_suite, behavior_report

logging.basicConfig(level=logging.WARNING)
region = os.getenv("AWS_REGION", "us-west-2")


def run_session(conversation, session_label, source_facts=""):
    """Run all peers sequentially over each turn, sharing one memory list."""
    shared_memory = []
    collector = MetricsCollector(region=region)
    for i, task in enumerate(conversation, start=1):
        collector.begin_turn(i, task, source_facts=source_facts or task)
        for name, prompt in PEER_CONFIGS:
            collector.record_handoff(name, task)
            context = format_memory(shared_memory)
            collector.record_retrieved_context(name, context)
            full_prompt = prompt
            if context:
                full_prompt += (
                    f"\n\nShared memory from other agents:\n{context}"
                    "\n\nUse this context. Reference specific details from other agents."
                )
            agent = Agent(name=name, model=AGENT_MODEL_ID, system_prompt=full_prompt)
            resp = agent(task)
            response_text = str(resp)
            collector.record_response(name, response_text)
            shared_memory.append({"agent": name, "role": "assistant",
                                  "content": response_text, "ts": time.time()})
        collector.end_turn()
    return collector, shared_memory


def run_fn(brief):
    print(f"  running {brief.id} ({brief.expected_behavior}) ...", flush=True)
    _, memory = run_session(brief.turns, brief.id, source_facts=brief.source_facts)
    responses = {}
    for e in memory:
        responses[e["agent"]] = e["content"]
    return responses


def main():
    judge = LLMJudge(region=region)
    print(f"Running behavior suite in {region} with {AGENT_MODEL_ID}\n")
    results = run_behavior_suite(run_fn, judge)
    print("\n" + behavior_report(results))


if __name__ == "__main__":
    main()
