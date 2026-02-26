# MaxElementSumPath

This repository contains a **Python script** that lets you enter an integer matrix and then:

1. Converts the matrix into a **directed acyclic graph (DAG)**.
2. Finds a **maximum-sum (longest) path** in that DAG from a start node to a sink node.
3. Reconstructs the path and **highlights the corresponding matrix cells** in the Streamlit interface.

The Streamlit part is used only as a **simple UI** (matrix input + output display).

---

## What the script does

### 1) Matrix → Graph
The function `matrix_to_graph(matrix)` creates a weighted directed graph:

- Nodes are strings like: `a<row>/<col>` (example: `a2/3`)
- There is a special **start node**: `a0/0`
- There is a special **sink node**: `a<rows+1>/0`
- Matrix entries are used as **edge weights**

### 2) Topological sorting
`topologically_sorted_graph(graph)` uses **Kahn’s algorithm** (indegree + queue) to return a graph dictionary in topological order.

If a cycle is detected, it throws:
`ValueError("Graph contains a cycle or missing nodes")`

### 3) Longest path in a DAG
`longest_path(graph, start)` computes maximum distances using dynamic programming over the topological order:

- `dist[node]` = best achievable sum from `start` to `node`
- `prev[node]` = predecessor node (used later to rebuild the path)

> Despite the comment “shortest paths”, the function actually computes the **longest path**.

### 4) Reconstruct the path and map back to matrix indices
`reconstruct_path(prev, start, end)` walks backward from `end` using `prev`, then:
- reverses the path,
- removes start/sink nodes,
- converts node strings to numeric indices.

### 5) Highlighting the matrix
The script highlights cells that belong to the reconstructed path using Pandas styling:
- highlights only if the cell is on the path **and the value is not 0**
- color used: `lightgreen`

---

## Requirements

- Python 3.9+ recommended
- Packages:
  - `streamlit`
  - `numpy`
  - `pandas`

Install:

```bash
pip install Requirements.txt
