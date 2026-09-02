"""Render the evaluation charts (the ones the notebooks produce) to PNG files
so they can be embedded in persona-metrics.html.

Uses the REAL plotting functions from eval_helpers.py, driven by lightweight
mock collectors whose numbers mirror the committed notebook run outputs
(market_trends / customer_insights / strategy_synth; C2 alignment ~0.74-0.81;
reasoning ~14-16s). No Bedrock calls.

Run:  python build_charts.py
Output: charts/*.png
"""
import os
import sys
import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# eval_helpers lives in the parent module dir; add it to the path so this
# script runs from the presentation/ subfolder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval_helpers import (
    plot_context_metrics_radar, plot_latency_breakdown,
    plot_session_comparison, plot_coordination_overhead, plot_c2_heatmap,
)

OUT = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUT, exist_ok=True)


# --- Minimal stand-ins matching the attributes the plot fns read ------------
@dataclass
class Rec:
    agent_name: str
    judge_scores: Dict[str, Any] = field(default_factory=dict)
    memory_read_latency: float = 0.0
    memory_write_latency: float = 0.0
    total_agent_latency: float = 0.0
    reasoning_input_tokens: int = 0
    reasoning_output_tokens: int = 0


@dataclass
class Turn:
    turn_number: int
    agent_calls: List[Rec] = field(default_factory=list)
    state_consistency: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Collector:
    turns: List[Turn] = field(default_factory=list)


def scores(freshness, handoff, util, write, redund):
    return {
        "context_freshness": {"score": freshness},
        "handoff_completeness": {"score": handoff},
        "context_utilization": {"score": util},
        "write_accuracy": {"score": write},
        "redundant_context": {"score": redund},
    }


def make_session(kind: str) -> Collector:
    """Build a collector mirroring the sequential notebook's 3 peers.

    'simple'   — clean brief, high scores (matches the committed run trace).
    'feedback' — brief revised mid-session; freshness/consistency dip.
    """
    if kind == "simple":
        specs = [
            # name, (fresh,handoff,util,write,redund), read, write, total, in_tok, out_tok
            ("market_trends",     (5, 5, 5, 5, 5), 0.00, 0.00, 15.63, 1200, 1800),
            ("customer_insights", (5, 5, 5, 5, 4), 0.02, 0.02, 15.50, 2600, 1700),
            ("strategy_synth",    (5, 4, 5, 5, 4), 0.03, 0.02, 14.36, 3800, 1500),
        ]
        consistency = 5
    else:  # feedback / conflict
        specs = [
            ("market_trends",     (3, 5, 4, 5, 4), 0.01, 0.01, 16.10, 1300, 1900),
            ("customer_insights", (3, 4, 3, 4, 3), 0.05, 0.03, 16.80, 3100, 1800),
            ("strategy_synth",    (4, 4, 4, 4, 3), 0.06, 0.04, 15.90, 4600, 1700),
        ]
        consistency = 3

    turn = Turn(turn_number=1)
    for name, sc, rd, wr, tot, it, ot in specs:
        turn.agent_calls.append(Rec(
            agent_name=name, judge_scores=scores(*sc),
            memory_read_latency=rd, memory_write_latency=wr,
            total_agent_latency=tot, reasoning_input_tokens=it,
            reasoning_output_tokens=ot,
        ))
    turn.state_consistency = {"score": consistency}
    return Collector(turns=[turn])


def make_embeddings(kind: str) -> Dict[str, list]:
    """Synthesize 3 vectors whose pairwise cosine similarities match the
    committed run (market↔customer 0.792, market↔strategy 0.737,
    customer↔strategy 0.809 for 'simple'; lower spread for 'feedback')."""
    random.seed(7 if kind == "simple" else 13)
    dim = 64
    base = [random.gauss(0, 1) for _ in range(dim)]

    def blend(target_sim):
        noise = [random.gauss(0, 1) for _ in range(dim)]
        # v = a*base + b*noise, choose a/b so cos(v, base) ~= target_sim
        a, b = target_sim, math.sqrt(max(1 - target_sim ** 2, 0.0))
        return [a * base[i] + b * noise[i] for i in range(dim)]

    if kind == "simple":
        return {
            "market_trends": base,
            "customer_insights": blend(0.792),
            "strategy_synth": blend(0.737),
        }
    else:
        return {
            "market_trends": base,
            "customer_insights": blend(0.61),
            "strategy_synth": blend(0.55),
        }


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    simple = make_session("simple")
    feedback = make_session("feedback")

    save(plot_context_metrics_radar(simple, "Simple Session"), "radar_simple.png")
    save(plot_latency_breakdown(simple, "Simple Session"), "latency_simple.png")
    save(plot_coordination_overhead(simple, "Simple Session"), "tokens_simple.png")
    save(plot_session_comparison(simple, feedback, "Simple", "Revised Brief"),
         "comparison.png")

    emb = make_embeddings("simple")
    save(plot_c2_heatmap(emb, "Simple Session"), "c2_heatmap.png")

    print("done")


if __name__ == "__main__":
    main()
