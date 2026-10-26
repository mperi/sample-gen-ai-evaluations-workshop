"""Generate the Agent Health Evaluation Demo architecture diagram.

Writes ``architecture.png`` (raster, for the README / blog) and
``architecture.svg`` (vector) next to this script.

Requirements: only the Graphviz binaries on PATH (``brew install graphviz``
or ``apt-get install graphviz``). No extra Python packages: the DOT source is
built as a string and rendered with ``neato -n2``.

Why ``neato -n2`` and pinned coordinates instead of plain ``dot``: the
diagram is organised in horizontal bands (AWS services on top, the pipeline
in the middle, the shared ``output/`` store as a "bus", the human review
step below it, and the Phase 5 loop along the bottom). ``dot`` reorders
nodes inside a rank to satisfy cluster contiguity, which breaks that band
order. Pinning every node and cluster keeps the layout stable and makes it
easy to adjust: edit the coordinates in ``LAYOUT`` / ``CLUSTERS`` below
(units are inches, origin bottom-left).

Usage:
    python docs/build_architecture_diagram.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
PNG_PATH = DOCS_DIR / "architecture.png"
SVG_PATH = DOCS_DIR / "architecture.svg"

PNG_DPI = 130          # ~4300 px wide for a 33 in canvas
FONT = "Helvetica"
PT = 72.0              # points per inch (Graphviz pos/bb units)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
LOCAL_FILL, LOCAL_LINE = "#EAF2FB", "#2F5D8A"   # local Python components
AWS_FILL, AWS_LINE = "#FFEBCC", "#C25E00"       # AWS managed services
AWS_BAND = "#FFF8EE"
HUMAN_FILL, HUMAN_LINE = "#E3F4E6", "#2E7D3E"   # domain expert
STORE_FILL, STORE_LINE = "#F2F2F2", "#5A5A5A"   # local artifact store
RO_FILL, RO_LINE = "#FBF3F3", "#9B1C1C"         # read-only source module
LOOP = "#7B3FA0"                                 # Phase 5 feedback loop
FUTURE = "#A0A0A0"                               # Part 2 placeholder
INK, MUTED = "#222222", "#5F5F5F"

# ---------------------------------------------------------------------------
# Geometry (inches). Canvas is CANVAS_W x CANVAS_H.
# ---------------------------------------------------------------------------
CANVAS_W, CANVAS_H = 33.2, 17.2
# Row centre lines
Y_AWS, Y_ROW2, Y_BUS, Y_ROW4 = 14.1, 10.9, 7.75, 3.5
Y_RECORDS = 9.35     # gap between row 2 and the output/ bus
Y_LOOP = 1.6         # Phase 5 loop runs along the bottom
LAYOUT = {
    # node id: (x, y)
    "title": (16.6, 16.45),
    # Read-only source module (left)
    "src_cfg": (3.5, 11.6),
    "src_cases": (3.5, Y_RECORDS),
    "expert": (3.5, Y_ROW4),
    # AWS band (each service sits above the component that calls it)
    "bedrock": (18.2, Y_AWS),
    "agentcore": (23.6, Y_AWS),
    "cloudwatch": (29.8, Y_AWS),
    # Row 2: Phase 1 + Phases 3-4
    "deriver": (12.0, Y_ROW2),
    "proposer": (18.2, Y_ROW2),
    "runner": (23.6, Y_ROW2),
    "publisher": (29.8, Y_ROW2),
    # output/ bus
    "o_contracts": (12.0, Y_BUS),
    "o_metrics": (18.2, Y_BUS),
    "o_feedback": (24.4, Y_BUS),
    "o_results": (28.0, Y_BUS),
    # Row 4: Phase 2 (+ Phase 5 view)
    "review": (14.8, Y_ROW4),
    "fstore": (24.4, Y_ROW4),
    "part2": (30.6, Y_ROW4),
    # Legend
    "legend": (16.6, 0.5),
    # Invisible waypoints for the long routes
    "wp_rec_a": (20.75, Y_RECORDS),     # records -> runner
    "wp_rev_a": (7.8, 8.55),            # real outputs -> review (left gutter)
    "wp_rev_b": (7.8, 4.0),
    "wp_run_w": (21.05, 10.6),          # arrow tips just outside the runner box
    "wp_run_s": (23.0, 9.93),
    "wp_run_se": (25.6, 9.93),
    "wp_run_e": (26.15, 10.9),
    "wp_loop_a": (28.0, Y_LOOP),        # Phase 5: results -> review
    "wp_loop_b": (14.8, Y_LOOP),
}
# Edge labels are drawn as separate plaintext nodes pinned beside their line
# (above a horizontal segment, to one side of a vertical/diagonal one) rather
# than as edge ``label``s, which neato -n2 centres directly on the spline.
EDGE_LABELS = {
    # label node id: (x, y)
    "lbl_prompts": (7.8, 11.62),        # src_cfg -> deriver (above, in the gutter)
    "lbl_records": (15.1, 9.62),        # src_cases -> runner (above the line)
    "lbl_outputs": (10.35, 4.14),       # src_cases -> review (above last segment)
    "lbl_expert": (6.9, 3.22),          # expert -> review (below the line)
    "lbl_contract": (15.35, 11.23),     # deriver -> proposer (above)
    "lbl_append": (20.3, 3.8),         # review -> fstore (above)
    "lbl_expected": (23.3, 8.61),       # feedback/ -> runner (left of the line)
    "lbl_evaluate": (22.75, 12.15),     # runner <-> AgentCore (left of the line)
    "lbl_scores": (27.7, 9.45),         # runner -> results/ (right of the line)
    "lbl_putmetric": (31.0, 12.1),      # publisher <-> CloudWatch (right of the line)
    "lbl_loop": (21.4, 1.88),           # Phase 5 bottom loop (above)
    "lbl_refine": (18.3, 5.58),         # review -> metrics/ (right of the line)
}
CLUSTERS = {
    # cluster id: (x0, y0, x1, y1)
    "cluster_source": (0.4, 8.55, 6.6, 13.35),
    "cluster_aws": (9.0, 13.05, 32.8, 15.55),
    "cluster_p1": (9.0, 9.9, 20.5, 12.6),
    "cluster_p34": (21.0, 9.9, 32.8, 12.6),
    "cluster_out": (9.0, 5.95, 32.8, 9.0),
    "cluster_p2": (9.0, 2.4, 26.5, 5.2),
}
def pos(node: str) -> str:
    x, y = LAYOUT[node]
    return f'pos="{x * PT:.1f},{y * PT:.1f}!"'


def bb(cluster: str) -> str:
    x0, y0, x1, y1 = CLUSTERS[cluster]
    return f'bb="{x0 * PT:.1f},{y0 * PT:.1f},{x1 * PT:.1f},{y1 * PT:.1f}"'


def label(title: str, *lines: str, file: str | None = None,
          title_size: int = 24, body_size: int = 20) -> str:
    """HTML-like node label: bold title, optional file name, short lines."""
    parts = [f'<B><FONT POINT-SIZE="{title_size}">{title}</FONT></B>']
    if file:
        parts.append(f'<FONT POINT-SIZE="18" COLOR="{MUTED}"><I>{file}</I></FONT>')
    parts += [f'<FONT POINT-SIZE="{body_size}">{line}</FONT>' for line in lines]
    return "<" + "<BR/>".join(parts) + ">"


def edge_label(node: str, text: str, color: str, html: bool = False) -> str:
    """Plaintext node with a white backing, pinned at EDGE_LABELS[node]."""
    x, y = EDGE_LABELS[node]
    lab = f"<{text}>" if html else f'"{text}"'
    return (f'{node} [shape=plaintext, style=filled, fillcolor="white", width=0, height=0, '
            f'margin="0.06,0.03", fontsize=19, fontcolor="{color}", '
            f'pos="{x * PT:.1f},{y * PT:.1f}!", label={lab}];')
def cluster_label(title: str, sub: str | None = None, color: str = INK) -> str:
    out = f'<B><FONT POINT-SIZE="25" COLOR="{color}">{title}</FONT></B>'
    if sub:
        out += f'<BR/><FONT POINT-SIZE="19" COLOR="{MUTED}">{sub}</FONT>'
    return f"<{out}>"


def legend_label() -> str:
    def swatch(fill: str, line: str, text: str, dashed: bool = False) -> str:
        style = ' STYLE="dashed"' if dashed else ""
        return (f'<TD BGCOLOR="{fill}" COLOR="{line}" BORDER="2" WIDTH="38" '
                f'HEIGHT="24"{style}> </TD>'
                f'<TD ALIGN="LEFT"><FONT POINT-SIZE="17">{text}</FONT></TD>'
                '<TD WIDTH="22"> </TD>')

    cells = "".join([
        swatch(LOCAL_FILL, LOCAL_LINE, "Local Python component"),
        swatch(AWS_FILL, AWS_LINE, "AWS managed service"),
        swatch(HUMAN_FILL, HUMAN_LINE, "Human"),
        swatch(STORE_FILL, STORE_LINE, "Local artifact (output/)"),
        swatch(RO_FILL, RO_LINE, "Read-only source", dashed=True),
        f'<TD><FONT POINT-SIZE="20" COLOR="{LOOP}"><B>&#8594;</B></FONT></TD>'
        f'<TD ALIGN="LEFT"><FONT POINT-SIZE="17" COLOR="{LOOP}">Phase 5 feedback loop</FONT></TD>',
        '<TD WIDTH="22"> </TD>',
        f'<TD><FONT POINT-SIZE="20" COLOR="{AWS_LINE}"><B>&#8596;</B></FONT></TD>'
        f'<TD ALIGN="LEFT"><FONT POINT-SIZE="17" COLOR="{AWS_LINE}">AWS API call</FONT></TD>',
    ])
    return f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4"><TR>{cells}</TR></TABLE>>'


def build_dot() -> str:
    W, H = CANVAS_W * PT, CANVAS_H * PT
    wp = 'shape=point, width=0.01, height=0.01, label="", style=invis'
    ro_edge = f'style=dashed, color="{RO_LINE}", fontcolor="{RO_LINE}", penwidth=2'
    aws_edge = (f'dir=both, arrowtail=normal, style=dashed, color="{AWS_LINE}", '
                f'fontcolor="{AWS_LINE}", penwidth=2.2')
    store_edge = f'color="{STORE_LINE}", penwidth=2'
    loop_edge = f'color="{LOOP}", fontcolor="{LOOP}", penwidth=3.2'

    return f"""
digraph AgentHealthEvaluation {{
  graph [bb="0,0,{W:.0f},{H:.0f}", splines=true, bgcolor="white",
         fontname="{FONT}", pad=0.3, outputorder=edgesfirst];
  node  [shape=box, style="rounded,filled", fontname="{FONT}", fontcolor="{INK}",
         margin="0.25,0.14", penwidth=2, color="{LOCAL_LINE}", fillcolor="{LOCAL_FILL}"];
  edge  [fontname="{FONT}", fontsize=19, penwidth=2, color="#444444", arrowsize=1.1];

  title [shape=plain, {pos("title")}, label=<<B><FONT POINT-SIZE="34">Agent Health Evaluation Demo · Architecture (Part 1)</FONT></B><BR/><FONT POINT-SIZE="20" COLOR="{MUTED}">From an existing agent's system prompt to health metrics, expert feedback, Amazon Bedrock AgentCore scores, and Amazon CloudWatch</FONT>>];

  // ---------------- Source agent module (read-only) ----------------
  subgraph cluster_source {{
    {bb("cluster_source")};
    label={cluster_label("Source agent module · READ-ONLY", "Multiagent Shared Context Evaluation/", RO_LINE)};
    labelloc=t; style="dashed,filled,rounded"; fillcolor="{RO_FILL}"; color="{RO_LINE}"; penwidth=3;
    src_cfg [{pos("src_cfg")}, fillcolor="white", color="{RO_LINE}",
             label={label("model_config.py", "PEER_CONFIGS system prompts:", "market_trends · customer_insights", "strategy_synth")}];
    src_cases [{pos("src_cases")}, fillcolor="white", color="{RO_LINE}",
               label={label("review/review_cases.jsonl", "real agent outputs")}];
  }}

  expert [{pos("expert")}, style="rounded,filled", fillcolor="{HUMAN_FILL}", color="{HUMAN_LINE}", penwidth=3,
          label={label("Domain expert", "reads real outputs,", "labels and judges them")}];

  // ---------------- AWS managed services ----------------
  subgraph cluster_aws {{
    {bb("cluster_aws")};
    label={cluster_label("AWS managed services", None, AWS_LINE)};
    labelloc=t; labeljust=l; style="rounded,filled"; fillcolor="{AWS_BAND}"; color="{AWS_LINE}"; penwidth=2.5;
    node [fillcolor="{AWS_FILL}", color="{AWS_LINE}", penwidth=2.5];
    bedrock [{pos("bedrock")}, label={label("Amazon Bedrock", "Claude Sonnet 4.5 · InvokeModel")}];
    agentcore [{pos("agentcore")}, label={label("Amazon Bedrock AgentCore", "Evaluate API · Builtin.Correctness", "synthetic single-span trace")}];
    cloudwatch [{pos("cloudwatch")}, label={label("Amazon CloudWatch", "namespace AgentHealthDemo", "dims: AgentName, MetricName")}];
  }}

  // ---------------- Phase 1 ----------------
  subgraph cluster_p1 {{
    {bb("cluster_p1")};
    label={cluster_label("Phase 1", "derive contract &amp; propose metrics")};
    labelloc=t; labeljust=l; style="rounded"; color="{LOCAL_LINE}"; penwidth=2;
    deriver [{pos("deriver")}, label={label("Contract Deriver", "system prompt &#8594; AgentContract", "role · responsibilities · constraints", "LLM + deterministic fallback", file="contract_deriver.py")}];
    proposer [{pos("proposer")}, label={label("Metric Proposer", "name · definition · rubric", "traced to a contract line", "invalid proposals rejected", file="metric_proposer.py")}];
  }}

  // ---------------- Phases 3-4 ----------------
  subgraph cluster_p34 {{
    {bb("cluster_p34")};
    label={cluster_label("Phases 3–4", "score &amp; publish · CLI entry point: run_pipeline.py")};
    labelloc=t; style="rounded"; color="{LOCAL_LINE}"; penwidth=2;
    runner [{pos("runner")}, label={label("Phase 3 · Evaluator Runner", "every (record, metric) pair", "agentcore_scorer", "per-pair failure isolation", file="evaluator_runner.py")}];
    publisher [{pos("publisher")}, label={label("Phase 4 · Dashboard Publisher", "aggregate score per metric", "publish_aggregates()", file="dashboard_publisher.py")}];
  }}

  // ---------------- Local artifact store (bus) ----------------
  subgraph cluster_out {{
    {bb("cluster_out")};
    label={cluster_label("Local artifact store · output/", "all writes via persistence.resolve_output_path")};
    labelloc=b; labeljust=l; style="rounded,filled"; fillcolor="{STORE_FILL}"; color="{STORE_LINE}"; penwidth=2;
    node [shape=cylinder, style="filled", fillcolor="white", color="{STORE_LINE}", margin="0.2,0.08"];
    o_contracts [{pos("o_contracts")}, label={label("contracts/", "&lt;agent&gt;.json", title_size=21, body_size=18)}];
    o_metrics [{pos("o_metrics")}, label={label("metrics/", "candidate_metrics.jsonl", title_size=21, body_size=18)}];
    o_feedback [{pos("o_feedback")}, label={label("feedback/", "feedback_store.jsonl", title_size=21, body_size=18)}];
    o_results [{pos("o_results")}, label={label("results/", "metric_results.jsonl", title_size=21, body_size=18)}];
  }}

  // ---------------- Phase 2 (+ Phase 5 view) ----------------
  subgraph cluster_p2 {{
    {bb("cluster_p2")};
    label={cluster_label("Phase 2", "expert review · the UI also shows Phase 5 results")};
    labelloc=t; labeljust=l; style="rounded"; color="{LOCAL_LINE}"; penwidth=2;
    review [{pos("review")}, label={label("Review UI (Streamlit)", "Expected State Label", "Task Success Feedback: pass/fail + rationale", "per-record scores · disagreement flags", file="review_app.py")}];
    fstore [{pos("fstore")}, label={label("Feedback Store", "append-only JSONL", "find_disagreements()", file="feedback_store.py")}];
  }}

  part2 [{pos("part2")}, style="rounded,dashed", color="{FUTURE}", fillcolor="white", fontcolor="{FUTURE}", penwidth=2,
         label=<<B><FONT POINT-SIZE="19">Part 2 (coming next)</FONT></B><BR/><FONT POINT-SIZE="17">golden baseline + regression gate</FONT>>];

  legend [shape=plain, {pos("legend")}, label={legend_label()}];

  // Waypoints (invisible)
  wp_rec_a [{wp}, {pos("wp_rec_a")}];
  wp_rev_a [{wp}, {pos("wp_rev_a")}];
  wp_rev_b [{wp}, {pos("wp_rev_b")}];
  wp_run_w [{wp}, {pos("wp_run_w")}];
  wp_run_s [{wp}, {pos("wp_run_s")}];
  wp_run_se [{wp}, {pos("wp_run_se")}];
  wp_run_e [{wp}, {pos("wp_run_e")}];
  wp_loop_a [{wp}, {pos("wp_loop_a")}];
  wp_loop_b [{wp}, {pos("wp_loop_b")}];
  // Edge labels (pinned plaintext nodes beside their lines)
  {edge_label("lbl_prompts", "system prompts", RO_LINE)}
  {edge_label("lbl_records", "records (read-only)", RO_LINE)}
  {edge_label("lbl_outputs", "real outputs", RO_LINE)}
  {edge_label("lbl_expert", "labels + feedback", HUMAN_LINE)}
  {edge_label("lbl_contract", "AgentContract", "black")}
  {edge_label("lbl_append", "append", "black")}
  {edge_label("lbl_expected", "expected-state labels", "black")}
  {edge_label("lbl_evaluate", "evaluate()", AWS_LINE)}
  {edge_label("lbl_scores", "scores", "black")}
  {edge_label("lbl_putmetric", "PutMetricData", AWS_LINE)}
  {edge_label("lbl_loop", "<B>Phase 5</B> · per-record scores + disagreement flags", LOOP, html=True)}
  {edge_label("lbl_refine", "<B>Phase 5</B> · refine metrics", LOOP, html=True)}

  // ================= Edges =================
  // Read-only inputs from the source module
  src_cfg -> deriver [{ro_edge}];
  src_cases -> wp_rec_a [{ro_edge}, arrowhead=none];
  wp_rec_a -> wp_run_w [{ro_edge}];
  src_cases -> wp_rev_a [{ro_edge}, arrowhead=none];
  wp_rev_a -> wp_rev_b [{ro_edge}, arrowhead=none];
  wp_rev_b -> review [{ro_edge}];

  // Phase 1
  deriver -> proposer [penwidth=2.6];
  deriver -> bedrock [{aws_edge}];
  proposer -> bedrock [{aws_edge}];
  deriver -> o_contracts [{store_edge}];
  proposer -> o_metrics [{store_edge}];

  // Phase 2
  expert -> review [color="{HUMAN_LINE}", fontcolor="{HUMAN_LINE}", penwidth=3];
  review -> fstore [penwidth=2.6];
  fstore -> o_feedback [{store_edge}];

  // Phase 3
  o_metrics -> wp_run_s [{store_edge}];
  o_feedback -> wp_run_se [{store_edge}];
  runner -> agentcore [{aws_edge}];
  runner -> wp_run_e [{store_edge}, arrowhead=none];
  wp_run_e -> o_results [{store_edge}];

  // Phase 4
  o_results -> publisher [{store_edge}];
  publisher -> cloudwatch [{aws_edge}];

  // Phase 5 feedback loop
  o_results -> wp_loop_a [{loop_edge}, arrowhead=none];
  wp_loop_a -> wp_loop_b [{loop_edge}, arrowhead=none];
  wp_loop_b -> review [{loop_edge}];
  review -> o_metrics [{loop_edge}];

  // Part 2 (secondary)
  o_results -> part2 [style=dashed, color="{FUTURE}", arrowhead=empty, penwidth=2];
}}
"""


def render(dot_src: str, fmt: str, out: Path, extra: list[str] | None = None) -> None:
    cmd = ["neato", "-n2", f"-T{fmt}", *(extra or []), "-o", str(out)]
    subprocess.run(cmd, input=dot_src, text=True, check=True)


def main() -> int:
    if shutil.which("neato") is None:
        print("error: Graphviz 'neato' not found on PATH (install graphviz)", file=sys.stderr)
        return 1
    dot_src = build_dot()
    if "--dot" in sys.argv:
        print(dot_src)
        return 0
    render(dot_src, "png", PNG_PATH, [f"-Gdpi={PNG_DPI}"])
    render(dot_src, "svg", SVG_PATH)
    print(f"wrote {PNG_PATH}")
    print(f"wrote {SVG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
