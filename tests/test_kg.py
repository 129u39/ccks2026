"""测试 KG 模块"""
import sys; sys.path.insert(0, "..")
from src.kg.parser import parse_triples
from src.kg.graph_builder import build_graph, build_graph_from_triples
from src.kg.graph_reasoner import bfs_search, dfs_search, path_search


def test_parse_triples():
    triples = parse_triples("A rel.action B | C rel.other D")
    assert len(triples) == 2
    assert triples[0] == ("A", "rel.action", "B")
    assert triples[1] == ("C", "rel.other", "D")


def test_parse_triples_complex_relation():
    text = "United States location.location.contains New York"
    triples = parse_triples(text)
    assert len(triples) == 1
    h, r, t = triples[0]
    assert h == "United States"
    assert r == "location.location.contains"
    assert t == "New York"


def test_build_graph():
    G = build_graph("A rel.friend B | B rel.friend C")
    assert G.has_edge("A", "B")
    assert G.has_edge("B", "C")
    assert not G.has_edge("A", "C")
    data = G.get_edge_data("A", "B")
    assert data[0]["relation"] == "rel.friend"


def test_build_graph_from_triples():
    triples = [("X", "likes", "Y")]
    G = build_graph_from_triples(triples)
    assert G.has_edge("X", "Y")


def test_bfs_search():
    G = build_graph("A rel.friend B | B rel.friend C")
    results = bfs_search(G, "A", max_hops=2)
    assert len(results) >= 2
    depths = [r["depth"] for r in results]
    assert 1 in depths
    assert 2 in depths


def test_dfs_search():
    G = build_graph("A rel.friend B | B rel.friend C")
    results = dfs_search(G, "A", max_hops=2)
    assert len(results) >= 2


def test_path_search():
    G = build_graph("A rel.friend B | B rel.friend C")
    paths = path_search(G, "A", "C", max_hops=3)
    assert len(paths) == 1


if __name__ == "__main__":
    test_parse_triples()
    test_parse_triples_complex_relation()
    test_build_graph()
    test_build_graph_from_triples()
    test_bfs_search()
    test_dfs_search()
    test_path_search()
    print("All KG tests passed")
