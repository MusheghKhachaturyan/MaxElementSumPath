import numpy as np
import streamlit as st
import pandas as pd
from collections import defaultdict, deque


# Поиск кратчайших путей
def longest_path(graph, start):
    # Initialize distances: -inf means unreachable, except start = 0
    dist = {node: float('-inf') for node in graph}
    prev = {node: None for node in graph}
    dist[start] = 0

    # Topological order based on node dictionary order (assumes already sorted)
    order = list(graph.keys())

    for node in order:
        if dist[node] != float('-inf'):
            for neighbor, weight in graph[node]:
                if dist[neighbor] < dist[node] + weight:
                    dist[neighbor] = dist[node] + weight
                    prev[neighbor] = node
    return dist, prev



# Функция восстановления пути
def reconstruct_path(prev, start, end):
    path = []
    while end is not None:
        path.append(end)
        end = prev[end]

    path = path[::-1][1:-1]  # Remove first and last elements

    formatted_path = [[int(node.split('/')[0][1:]), int(node.split('/')[1])]
                      for node in path]  # Extract numbers
    return formatted_path


# Преобразование матрицы в граф
def topologically_sorted_graph(graph):
    # Step 1: Calculate indegree for each node
    indegree = defaultdict(int)
    all_nodes = set(graph.keys())

    for u in graph:
        for v, _ in graph[u]:
            indegree[v] += 1
            all_nodes.add(v)

    # Step 2: Initialize queue with nodes having zero indegree
    queue = deque([node for node in all_nodes if indegree[node] == 0])
    sorted_keys = []

    while queue:
        node = queue.popleft()
        sorted_keys.append(node)
        for neighbor, _ in graph.get(node, []):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    # Step 3: Validate and build sorted graph
    if len(sorted_keys) != len(all_nodes):
        raise ValueError("Graph contains a cycle or missing nodes")

    # Reconstruct dictionary in topological order
    sorted_graph = {node: graph.get(node, []) for node in sorted_keys}
    return sorted_graph


def matrix_to_graph(matrix):
    graph = {}
    rows, cols = len(matrix), len(matrix[0])
    num = cols - rows + 1
    max_element = max(map(max, matrix))

    graph['a0/0'] = []
    for i in range(cols):
        graph['a0/0'].append((f'a1/{i}', matrix[0][i]))
    for i in range(rows - 1):
        graph['a0/0'].append((f'a{i+2}/0', matrix[i + 1][0]))

    for i in range(rows - 1):
        for j in range(cols):
            for k in range(j, cols):
                node = f'a{i+1}/{j}'
                if node not in graph:
                    graph[node] = []
                if i + 1 < rows and k + 1 < cols:
                    graph[node].append((f'a{i+2}/{k+1}', matrix[i+1][k+1]))
            for k in range(i, rows):
                node = f'a{i+1}/{j}'
                if node not in graph:
                    graph[node] = []
                if k + 1 < rows and j + 1 < cols:
                    graph[node].append((f'a{k+2}/{j+1}', matrix[k+1][j+1]))

    for i in range(cols):
        graph[f'a{rows}/{i}'] = []
        graph[f'a{rows}/{i}'].append((f'a{rows+1}/0', 0))
    for i in range(rows - 1):
        graph[f'a{i+1}/{cols-1}'].append((f'a{rows+1}/0', 0))

    graph[f'a{rows+1}/0'] = []

    graph = topologically_sorted_graph(graph)

    return graph


st.title("Matrix To Path")

rows = st.number_input("Rows", min_value=1, step=1, value=3)
cols = st.number_input("Columns", min_value=1, step=1, value=4)

# Ввод целочисленной матрицы
default_matrix = pd.DataFrame(np.zeros((rows, cols), dtype=int))
matrix_input = st.data_editor(default_matrix, num_rows="dynamic", use_container_width=True)

try:
    matrix = matrix_input.to_numpy(dtype=int)

    graph = matrix_to_graph(matrix)
    start_node = 'a0/0'
    end_node = next(reversed(graph))
    distances, predecessors = longest_path(graph, start_node)
    shortest_path_to_end = reconstruct_path(predecessors, start_node, end_node)

    # Индексы пути
    highlight_indices = {(i - 1, j) for i, j in shortest_path_to_end if i > 0}


    def highlight_nonzero_path(x):
        df = pd.DataFrame('', index=x.index, columns=x.columns)
        for i in range(len(x.index)):
            for j in range(len(x.columns)):
                if (i, j) in highlight_indices and x.iat[i, j] != 0:
                    df.iat[i, j] = 'background-color: lightgreen'
        return df


    st.write("### Path:")
    styled = matrix_input.style.apply(highlight_nonzero_path, axis=None)
    st.dataframe(styled, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
