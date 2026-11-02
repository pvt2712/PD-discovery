"""knowledge — temporal background knowledge for causal-learn's PC and FCI, with the semantics of tiered FCI.

causal-learn turns tier knowledge into fully directed edges (X_t -> X_t+1), which asserts a tail and so rules out
latent confounding, and it refuses arrowheads at earlier-tier nodes. Temporal order only says that the later node is
not an ancestor of the earlier one: an arrowhead at the later node, the earlier node's mark left open (tFCI, Bang &
Didelez, CLeaR 2025). `install()` patches causal-learn's FCI with those semantics (from
ppmi-causal/src/ppmi_causal/discovery.py:_patch_fci) and adds the pooled analysis's two kinds of root nodes:

- context nodes (era2, genetic) have no causes: every edge at a context node has a tail there;
- exogenous nodes (age, sex, education, APOE e4, years since enrolment) may be caused by context nodes only.

The patch is module state of causal-learn, so `install()` is called before every search with that search's sets;
a search without knowledge installs empty sets (the negative-control runs must be free to violate them).
"""
import itertools

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


def install(context=(), exogenous=(), pds_depth=-1):
    """Install the tier semantics into causal-learn's FCI module for the given root-node sets; returns FCI.fci.
    `pds_depth` >= 0 caps the size of the separating sets FCI's possible-d-separation stage tries (PROTOCOL.md §5.1):
    causal-learn tries every subset of each edge's possible-d-sep set, which on the dense two-slice graphs at the
    pooled sample size does not finish (Fisher-z FCI ran > 67 CPU-minutes on one dataset)."""
    from causallearn.search.ConstraintBased import FCI

    context, exogenous = set(context), set(exogenous)
    FCI.removeByPossibleDsep = _capped_pds(FCI, pds_depth)

    def _set(graph, edge, src, dst, mark_src=None):
        graph.remove_edge(edge)
        graph.add_edge(Edge(src, dst, mark_src if mark_src is not None else edge.get_proximal_endpoint(src), Endpoint.ARROW))

    def _orient(bk, graph):
        if bk is None:
            return
        for edge in graph.get_graph_edges():
            n1, n2 = edge.get_node1(), edge.get_node2()
            a, b = n1.get_name(), n2.get_name()
            if (a in context) != (b in context):                  # context -> anything
                src, dst = (n1, n2) if a in context else (n2, n1)
                _set(graph, edge, src, dst, Endpoint.TAIL)
            elif a in context:                                    # two context nodes: left to the forbidden list
                continue
            elif (a in exogenous) != (b in exogenous):            # exogenous -> system node
                src, dst = (n1, n2) if a in exogenous else (n2, n1)
                _set(graph, edge, src, dst, Endpoint.TAIL)
            elif bk.is_forbidden(n1, n2):                         # n1 is in a later tier: arrowhead at n1
                _set(graph, edge, n2, n1)
            elif bk.is_forbidden(n2, n1):
                _set(graph, edge, n1, n2)

    def _arrow_allowed(node_x, node_y, graph, knowledge):
        y, x = node_y.get_name(), node_x.get_name()
        if y in context or (y in exogenous and x not in context):
            return False
        return graph.get_endpoint(node_x, node_y) in (Endpoint.ARROW, Endpoint.CIRCLE)

    FCI.fci_orient_bk = _orient
    FCI.is_arrow_point_allowed = _arrow_allowed
    return FCI.fci


def background(names, tiers, context=(), exogenous=(), node=None):
    """causal-learn BackgroundKnowledge from tiers, with no edges among context nodes or among exogenous nodes
    (independent by design: they are roots). `node` maps a name to the GraphNode the search will use."""
    bk = BackgroundKnowledge()
    nodes = {n: (node(n) if node else GraphNode(n)) for n in names}
    for k, tier in enumerate(tiers):
        for n in tier:
            if n in nodes:
                bk.add_node_to_tier(nodes[n], k)
    for group in (context, exogenous):
        for a, b in itertools.permutations([n for n in group if n in nodes], 2):
            bk.add_forbidden_by_node(nodes[a], nodes[b])
    return bk, nodes


def _capped_pds(FCI, pds_depth):
    """causallearn FCI.removeByPossibleDsep (0.1.4.8) with the subset size capped at `pds_depth` (-1: no cap)."""
    from causallearn.utils.DepthChoiceGenerator import DepthChoiceGenerator

    def remove(graph, independence_test_method, alpha, sep_sets):
        for edge in graph.get_graph_edges():
            for a, b in ((edge.get_node1(), edge.get_node2()), (edge.get_node2(), edge.get_node1())):
                if not graph.contains_edge(edge):
                    break
                pds = FCI.getPossibleDsep(a, b, graph, -1)
                cap = len(pds) if pds_depth < 0 else min(len(pds), pds_depth)
                adj_a = set(graph.get_adjacent_nodes(edge.get_node1()))
                adj_b = set(graph.get_adjacent_nodes(edge.get_node2()))
                gen = DepthChoiceGenerator(len(pds), cap)
                choice = gen.next()
                while choice is not None:
                    current, choice = choice, gen.next()
                    if len(current) < 2:
                        continue
                    sepset = [pds[i] for i in current]
                    if set(sepset) <= adj_a or set(sepset) <= adj_b:
                        continue
                    X, Y = graph.get_node_map()[edge.get_node1()], graph.get_node_map()[edge.get_node2()]
                    cond = tuple(graph.get_node_map()[n] for n in sepset)
                    if independence_test_method(X, Y, cond) > alpha:
                        graph.remove_edge(edge)
                        sep_sets[(X, Y)] = set(cond)
                        break

    return remove
