"""Guide §7.22-7.25: scripts/md_to_docx.py on a tiny Markdown file (synthetic only).

python-docx is not in the ppmi-causal environment, so run this with the system Python, as the
script's own docstring does. The Markdown and the Word file live in a temporary folder.

    /usr/bin/python3 docs/guide_examples/s07/ex_md_to_docx.py
"""
import sys
import tempfile
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import md_to_docx as M  # noqa: E402

MD = """# A tiny document
Intro with **bold**, *italic*, `code`, a [link](https://example.org) and 2 * 3 * 4 = 24.
A second line of the same paragraph; snake_case and $a*b*c$ stay as typed.

## Section one
- a bullet
  that continues on the next line
  - a nested bullet
1. a numbered item
2026. a sentence that starts with a year
> A note
> on two lines.

| Domain | Your answer |
|---|---|
| Tremor \\| rest | |
| | |

```
code line with **stars**
```
[ ] a task box
☐ an answer option
<!-- pagebreak -->
##### Deep heading
"""

tmp = Path(tempfile.mkdtemp())
(tmp / "tiny.md").write_text(MD, encoding="utf-8")
M.convert(tmp / "tiny.md", tmp / "tiny.docx", "Guide example · footer")
doc = Document(str(tmp / "tiny.docx"))


def fmt(run):
    tags = []
    if run.bold:
        tags.append("b")
    if run.italic:
        tags.append("i")
    if run.font.name == "Consolas":
        tags.append("mono")
    return f"{run.text!r}" + (f"[{','.join(tags)}]" if tags else "")


print("1. paragraphs (style: runs)")
for p in doc.paragraphs:
    if p.text.strip() or p.runs:
        breaks = p._p.xpath(".//w:br[@w:type='page']")
        print(f"   {p.style.name:<14} " + " + ".join(fmt(r) for r in p.runs) + ("  <page break>" if breaks else ""))
t = doc.tables[0]
print("2. table cells:", [[c.text for c in row.cells] for row in t.rows])
print("   column widths (cm):", [round(c.width.cm, 2) for c in t.columns])
print("   rows given a minimum height (empty rows are space to write in):",
      [r._tr.trPr is not None and r._tr.trPr.find(qn("w:trHeight")) is not None for r in t.rows])
print("3. footer:", doc.sections[0].footer.paragraphs[0].text, "+ a PAGE field")
print("4. core properties: title =", repr(doc.core_properties.title), "| author =", repr(doc.core_properties.author))
print("5. INLINE split of '2 * 3 * 4 and *x*':", [s for s in M.INLINE.split("2 * 3 * 4 and *x*") if s])
