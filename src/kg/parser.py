"""Stage 5: KG parser — 解析三元组"""

from __future__ import annotations


def parse_triples(text: str) -> list[tuple[str, str, str]]:
    """从输入文本中解析 (head, relation, tail) 三元组

    输入格式:
        "United States of America location.location.contains Manhattan Plaza Apartments I"
        "Balls to the Wall film.film.initial_release_date 2011"

    relation 始终包含点号（.），利用此特征切分三元组。
    """
    triples = []
    for line in text.split("|"):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 3:
            continue

        # 找到第一个带点号的 token —— 关系部分的开始
        rel_start = None
        for i, token in enumerate(parts):
            if "." in token:
                rel_start = i
                break

        if rel_start is None:
            # 兜底：全部当作 head
            continue

        # relation 从 rel_start 到最后一个带点号的 token
        rel_end = None
        for i in range(len(parts) - 1, rel_start - 1, -1):
            if "." in parts[i]:
                rel_end = i
                break

        if rel_end is None:
            continue

        head = " ".join(parts[:rel_start])
        relation = " ".join(parts[rel_start:rel_end + 1])
        tail = " ".join(parts[rel_end + 1:])

        triples.append((head.strip(), relation.strip(), tail.strip()))

    return triples
