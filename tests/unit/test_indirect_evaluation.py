"""La référence distingue une dépendance indirecte d'un lien déjà direct."""

from scripts.evaluate_integrated_evidence import indirect_pairs


def test_indirect_chain_keeps_acyclic_dependence_and_excludes_shortcut():
    assert indirect_pairs({("A", "B"), ("B", "C"), ("C", "D")}) == {
        ("A", "C"),
        ("A", "D"),
        ("B", "D"),
    }
    result = indirect_pairs({("A", "B"), ("B", "C"), ("A", "C")})
    assert ("A", "C") not in result


def test_a_missing_received_edge_does_not_create_an_observable_indirect_link():
    global_edges = {("A", "B"), ("B", "C"), ("X", "Y")}
    received = {("A", "B"), ("X", "Y")}
    assert indirect_pairs(global_edges) == {("A", "C")}
    assert indirect_pairs(received) == set()
