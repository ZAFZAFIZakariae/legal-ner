import sqlite3
import networkx as nx

def build_graph(db_path="legal_ner.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    G = nx.DiGraph()

    # 1) Add law/article nodes
    c.execute("SELECT law_id, normalized FROM laws")
    for lid, name in c.fetchall():
        G.add_node(f"law_{lid}", label=name, type="law")
    c.execute("SELECT article_id, law_id, normalized FROM articles")
    for aid, lid, name in c.fetchall():
        G.add_node(f"art_{aid}", label=name, type="article")
        # contains edge: law → article
        G.add_edge(f"law_{lid}", f"art_{aid}", rel="contains")

    # 2) Add mention nodes and referential edges
    c.execute("SELECT mention_id, entity_type, law_id, article_id FROM mentions")
    for mid, etype, lid, aid in c.fetchall():
        node = f"men_{mid}"
        # type: LAW or ARTICLE
        G.add_node(node, type=f"mention_{etype}")
        # normalize edge
        if lid:
            G.add_edge(node, f"law_{lid}", rel="maps_to")
        if aid:
            G.add_edge(node, f"art_{aid}", rel="maps_to")

    # 3) Inter-mention references (optional)
    # e.g., if a mention’s text contains “المادة X من القانون Y” link men_X → men_Y
    # You can query mentions text and apply regex to find such cross-links,
    # then add edges with rel='refers_to'.

    conn.close()
    return G
