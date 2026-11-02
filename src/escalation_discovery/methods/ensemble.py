"""ensemble — majority vote across method families (PROTOCOL.md §5.1, M1).

Each family votes once: a pair is adjacent in the family's vote when more than half of the family's members make it
adjacent. The ensemble keeps a pair when more than half of the families that produced a graph vote for it. Its mark
at each end is the majority mark among the members that kept the pair (arrowhead or tail when more than half agree,
otherwise a circle), so the output is read as a PAG.
"""
from collections import defaultdict

from ..graphs import MarkGraph


def majority(graphs: dict, families: dict, names):
    """graphs: method name -> MarkGraph; families: method name -> family."""
    fam = defaultdict(list)
    for m, g in graphs.items():
        if g is not None:
            fam[families[m]].append(g)
    votes = defaultdict(int)
    for members in fam.values():
        count = defaultdict(int)
        for g in members:
            for p in g.adjacencies():
                count[p] += 1
        for p, c in count.items():
            if c > len(members) / 2:
                votes[p] += 1
    out = MarkGraph(names, "PAG")
    nfam = len(fam)
    for p, v in votes.items():
        if v <= nfam / 2:
            continue
        a, b = sorted(p, key=out.index.get)
        marks = [g.marks(a, b) for members in fam.values() for g in members if a in g.index and g.adjacent(a, b)]
        res = []
        for k in (0, 1):
            arrows = sum(m[k] == "A" for m in marks)
            tails = sum(m[k] == "T" for m in marks)
            res.append("A" if arrows > len(marks) / 2 else "T" if tails > len(marks) / 2 else "C")
        out.add(a, b, *res)
    return out
