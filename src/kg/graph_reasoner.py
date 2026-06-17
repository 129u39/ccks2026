"""Stage 5: graph_reasoner — 图推理（BFS / DFS / Path Search）"""

from __future__ import annotations

import networkx as nx


def bfs_search(
    G: nx.MultiDiGraph,
    start_node: str,
    max_hops: int = 3,
) -> list[dict]:
    """BFS 遍历，返回可达节点及路径"""
    results = []
    visited = set()

    # queue: (current_node, path_edges, depth)
    queue = [(start_node, [], 0)]
    visited.add(start_node)

    while queue:
        current, path, depth = queue.pop(0)

        if depth > 0:
            results.append({
                "node": current,
                "path": path,
                "depth": depth,
            })

        if depth >= max_hops:
            continue

        for u, v, data in G.out_edges(current, data=True):
            neighbor = v
            if neighbor not in visited or depth < max_hops - 1:
                visited.add(neighbor)
                new_path = path + [(current, neighbor, data.get("relation", ""))]
                queue.append((neighbor, new_path, depth + 1))

    return results


def dfs_search(
    G: nx.MultiDiGraph,
    start_node: str,
    max_hops: int = 3,
) -> list[dict]:
    """DFS 遍历"""
    results = []

    def _dfs(current: str, path: list, depth: int):
        if depth > max_hops:
            return
        if depth > 0:
            results.append({
                "node": current,
                "path": path.copy(),
                "depth": depth,
            })
        for u, v, data in G.out_edges(current, data=True):
            new_path = path + [(current, v, data.get("relation", ""))]
            _dfs(v, new_path, depth + 1)

    _dfs(start_node, [], 0)
    return results


def path_search(
    G: nx.MultiDiGraph,
    source: str,
    target: str,
    max_hops: int = 3,
) -> list[list[tuple[str, str, str]]]:
    """查找两节点间所有路径（不超过 max_hops 跳）"""
    all_paths = []
    try:
        for path in nx.all_simple_paths(G, source=source, target=target, cutoff=max_hops):
            # 转换为 (node_from, node_to, relation) 格式
            edge_path = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                # 取第一条边的关系
                edge_data = G.get_edge_data(u, v)
                if edge_data:
                    first_key = list(edge_data.keys())[0]
                    relation = edge_data[first_key].get("relation", "")
                    edge_path.append((u, v, relation))
            all_paths.append(edge_path)
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        pass

    return all_paths
