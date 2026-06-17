"""Stage 5: graph_builder — 从三元组构建 NetworkX 图"""

from __future__ import annotations

import networkx as nx

from .parser import parse_triples


def build_graph(triple_text: str) -> nx.MultiDiGraph:
    """从三元组文本构建有向多重图"""
    triples = parse_triples(triple_text)
    G = nx.MultiDiGraph()

    for h, r, t in triples:
        G.add_edge(h, t, relation=r)

    return G


def build_graph_from_triples(triples: list[tuple[str, str, str]]) -> nx.MultiDiGraph:
    """从已解析的三元组列表构建图"""
    G = nx.MultiDiGraph()
    for h, r, t in triples:
        G.add_edge(h, t, relation=r)
    return G
