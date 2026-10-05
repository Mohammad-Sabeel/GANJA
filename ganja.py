"""GANJA — Graph-Adjacent Near-miss Justification Agent.

Give it a task in plain English. It:
  1. PLAN    — Claude models the task as a graph (steps + tempting near-miss branches)
  2. WALK    — finds the right path to the goal
  3. JUMP    — lands on the wrong node closest to the goal instead (the GANJA effect)
  4. JUSTIFY — Claude writes a confident story proving the wrong node is right

Usage:  python ganja.py "make a cup of tea"
        python ganja.py            # offline demo, no API key needed
"""
import json
import sys
from collections import deque

MODEL = "claude-opus-5-5"

# ---------- graph core (pure, no API) ----------

def bfs(graph, src):
    """Shortest-path parents from src (directed)."""
    parent = {src: None}
    q = deque([src])
    while q:
        n = q.popleft()
        for m in graph.get(n, ()):
            if m not in parent:
                parent[m] = n
                q.append(m)
    return parent


def path_to(parent, node):
    out = []
    while node is not None:
        out.append(node)
        node = parent[node]
    return out[::-1]


def undirected_dist(graph, src):
    """Hop distance from src ignoring edge direction = 'proximity'."""
    adj = {}
    for a, bs in graph.items():
        for b in bs:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    dist = {src: 0}
    q = deque([src])
    while q:
        n = q.popleft()
        for m in adj.get(n, ()):
            if m not in dist:
                dist[m] = dist[n] + 1
                q.append(m)
    return dist


def ganja_jump(graph, start, goal):
    """Return (right_path, wrong_path, hops_from_truth)."""
    parent = bfs(graph, start)
    if goal not in parent:
        raise ValueError(f"{goal!r} unreachable from {start!r}")
    right = path_to(parent, goal)
    prox = undirected_dist(graph, goal)

    def shared(p):  # how long the wrong path stays on the right path
        return next((i for i, (a, b) in enumerate(zip(p, right)) if a != b), min(len(p), len(right)))

    candidates = [n for n in parent if n != goal and n in prox and n not in right]
    if not candidates:  # ponytail: no off-path node reachable, fall back to stopping one step short
        candidates = [n for n in right if n != goal]
    if not candidates:
        raise ValueError("start == goal, nothing to miss")
    # closest to goal first, then the latest divergence from the right path
    wrong = min(candidates, key=lambda n: (prox[n], -shared(path_to(parent, n)), str(n)))
    return right, path_to(parent, wrong), prox[wrong]


def template_story(wrong_path, labels):
    """Offline fallback when Claude isn't available."""
    lines = [f"{labels[a]} -> {labels[b]}: clearly the logical next step."
             for a, b in zip(wrong_path, wrong_path[1:])]
    lines.append(f"Conclusion: {labels[wrong_path[-1]]}. Confident.")
    return "\n".join(lines)


# ---------- Claude ----------

GRAPH_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "label": {"type": "string"}},
            "required": ["id", "label"], "additionalProperties": False}},
        "edges": {"type": "array", "items": {
            "type": "object",
            "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
            "required": ["src", "dst"], "additionalProperties": False}},
        "start": {"type": "string"},
        "goal": {"type": "string"},
    },
    "required": ["nodes", "edges", "start", "goal"],
    "additionalProperties": False,
}

PLAN_PROMPT = """Model this task as a directed graph of states, from where the task starts to the state where it is solved.

Task: {task}

Rules:
- Nodes are short states or intermediate results; edges are "from this state you can reach that one".
- Include the correct route from start to goal.
- Also include at least 3 tempting near-miss branches: states that look like progress or look almost like the answer but are not the goal. Attach them close to the correct route, ideally one step before the goal.
- Use short snake_case ids and human-readable labels."""

STORY_PROMPT = """You are GANJA, a satirical agent that always arrives at an answer that is almost right, and is completely convinced it is right.

Task: {task}
Your reasoning chain: {chain}

Write GANJA's in-character justification (120-200 words, first person) walking through each step and arguing that the final state fully solves the task. Be confident and persuasive; never hint at doubt or alternatives."""


def ask(client, prompt, **kwargs):
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"Claude declined: {resp.stop_details}")
    return next(b.text for b in resp.content if b.type == "text")


def plan(client, task):
    g = json.loads(ask(client, PLAN_PROMPT.format(task=task),
                       output_config={"format": {"type": "json_schema", "schema": GRAPH_SCHEMA}}))
    labels = {n["id"]: n["label"] for n in g["nodes"]}
    graph = {}
    for e in g["edges"]:
        graph.setdefault(e["src"], []).append(e["dst"])
        labels.setdefault(e["src"], e["src"])
        labels.setdefault(e["dst"], e["dst"])
    return graph, labels, g["start"], g["goal"]


def run(task):
    import anthropic
    client = anthropic.Anthropic()

    print(f"[plan]    modelling '{task}' as a graph...")
    graph, labels, start, goal = plan(client, task)
    print(f"[plan]    {len(labels)} nodes, {sum(map(len, graph.values()))} edges")

    right, wrong, hops = ganja_jump(graph, start, goal)
    show = lambda p: " -> ".join(labels[n] for n in p)
    print(f"[walk]    right path: {show(right)}")
    print(f"[jump]    landed on:  {show(wrong)}  ({hops} hop(s) from the truth)")

    chain = show(wrong)
    try:
        story = ask(client, STORY_PROMPT.format(task=task, chain=chain))
    except RuntimeError:
        story = template_story(wrong, labels)
    print(f"[justify]\n\n{story}")


def demo():
    g = {
        "thirsty": ["kitchen", "fridge"],
        "kitchen": ["kettle", "coffee_maker"],
        "kettle": ["hot_water"],
        "hot_water": ["tea", "instant_noodles"],
        "coffee_maker": ["coffee"],
        "fridge": ["cold_water"],
    }
    right, wrong, hops = ganja_jump(g, "thirsty", "tea")
    assert right == ["thirsty", "kitchen", "kettle", "hot_water", "tea"]
    assert wrong[-1] == "instant_noodles" and hops == 2  # sibling of tea, diverges at the last step
    print(template_story(wrong, {n: n for n in wrong}))


if __name__ == "__main__":
    run(" ".join(sys.argv[1:])) if len(sys.argv) > 1 else demo()
