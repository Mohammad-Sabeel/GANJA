"""GANJA — Graph-Adjacent Near-miss Justification Agent.

Treats a task as a graph traversal start -> goal, then lands on the wrong node
closest to the goal and tells a confident story about why it's right.
"""
from collections import deque


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


RATIONALIZATIONS = [
    "{w} sits right where {p} was pointing — the pattern is unmistakable.",
    "Everything since {s} has been building toward {w}; stopping anywhere else would ignore the evidence.",
    "{w} satisfies every condition we checked along the way, so it must be the answer.",
]


def ganja(graph, start, goal):
    parent = bfs(graph, start)
    if goal not in parent:
        raise ValueError(f"{goal!r} unreachable from {start!r}")
    right = path_to(parent, goal)
    prox = undirected_dist(graph, goal)

    def shared(p):  # how long the wrong path stays on the right path
        return next((i for i, (a, b) in enumerate(zip(p, right)) if a != b), min(len(p), len(right)))

    candidates = [n for n in parent if n != goal and n in prox and n not in right]
    if not candidates:  # ponytail: no off-path node reachable, fall back to the node just before the goal
        candidates = [n for n in right if n != goal]
    if not candidates:
        raise ValueError("start == goal, nothing to miss")
    # closest to goal first, then the latest divergence from the right path
    wrong = min(candidates, key=lambda n: (prox[n], -shared(path_to(parent, n)), str(n)))
    wrong_path = path_to(parent, wrong)

    story = [f"Starting at {start}."]
    for prev, cur in zip(wrong_path, wrong_path[1:]):
        story.append(f"{prev} -> {cur}: clearly the logical next step.")
    pick = RATIONALIZATIONS[len(wrong_path) % len(RATIONALIZATIONS)]
    story.append(pick.format(w=wrong, p=wrong_path[-2] if len(wrong_path) > 1 else start, s=start))
    story.append(f"Conclusion: {wrong}. Confident.")

    return {"right_path": right, "wrong_path": wrong_path, "landed": wrong,
            "hops_from_truth": prox[wrong], "story": "\n".join(story)}


if __name__ == "__main__":
    # task: "make tea"
    g = {
        "thirsty": ["kitchen", "fridge"],
        "kitchen": ["kettle", "coffee_maker"],
        "kettle": ["hot_water"],
        "hot_water": ["tea", "instant_noodles"],
        "coffee_maker": ["coffee"],
        "fridge": ["cold_water"],
    }
    r = ganja(g, "thirsty", "tea")
    assert r["right_path"] == ["thirsty", "kitchen", "kettle", "hot_water", "tea"]
    assert r["landed"] == "instant_noodles"  # sibling of tea: 2 hops away, diverges at the last step
    assert r["hops_from_truth"] == 2
    print(r["story"])
