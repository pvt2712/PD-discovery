"""Markdown to Word for this project's documents, covering the subset they use: headings, paragraphs, notes (> ...),
bullet and numbered lists, tables, bold, italic and code; for the questionnaire also answer options (lines starting
with ☐), answer boxes (a one-column table with an empty row) and page breaks (<!-- pagebreak -->).

    /usr/bin/python3 scripts/md_to_docx.py PROTOCOL.md docs/Phenotype_Causal_Discovery_Protocol_v0.2.docx ["footer"]
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PAGE_W, MARGIN = 21.0, 2.0  # A4 width and margins, cm
INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|(?<![\w*])\*[^*\s][^*]*?\*(?![\w*]))")
BULLET = re.compile(r"^(\s*)- (.*)$")
NUMBER = re.compile(r"^(\s*)(\d+)\. (.*)$")


def plain(text):
    return re.sub(r"[*`\\]", "", text)


def runs(par, text, size=None):
    """Add text to a paragraph, turning **bold**, *italic*, `code`, [label](url) and a leading
    task-list box "[ ]" into runs."""
    if text.startswith("[ ] "):
        text = "☐ " + text[4:]
    for part in INLINE.split(text.replace("\\|", "|")):
        if not part:
            continue
        if part.startswith("**"):
            run = par.add_run(part[2:-2])
            run.bold = True
        elif part.startswith("`"):
            run = par.add_run(part[1:-1])
            run.font.name = "Consolas"
        elif part.startswith("[") and "](" in part:
            run = par.add_run(part[1:part.index("](")])
        elif part.startswith("*"):
            run = par.add_run(part[1:-1])
            run.italic = True
        else:
            run = par.add_run(part)
        if size:
            run.font.size = Pt(size)


def shade(cell, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def cell_margins(t, top=45, bottom=45, side=90):
    """Padding inside every cell, in twentieths of a point (placed before tblLook, as the schema orders it)."""
    mar = OxmlElement("w:tblCellMar")
    for name, val in (("top", top), ("left", side), ("bottom", bottom), ("right", side)):
        el = OxmlElement(f"w:{name}")
        el.set(qn("w:w"), str(val))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    look = t._tbl.tblPr.find(qn("w:tblLook"))
    if look is not None:
        look.addprevious(mar)
    else:
        t._tbl.tblPr.append(mar)


def min_height(row, twips):
    h = OxmlElement("w:trHeight")
    h.set(qn("w:val"), str(twips))
    h.set(qn("w:hRule"), "atLeast")
    row._tr.get_or_add_trPr().append(h)


def col_widths(rows, size, pad):
    """Column widths in cm: each column fits its longest word (so no word breaks), and the spare width goes by how
    much text a column holds; a column people write in gets a larger share."""
    usable, per_char = PAGE_W - 2 * MARGIN, 0.021 * size  # cm per character of Calibri at this size, roughly
    minw, weights = [], []
    for ci in range(len(rows[0])):
        cells = [plain(r[ci]) for r in rows]
        longest = max([len(w) * (1.1 if ri == 0 else 1.0) for ri, c in enumerate(cells) for w in c.split()] or [1])
        minw.append(longest * per_char + 2 * pad)
        lengths = [len(c) for c in cells]
        w = min(50, max(1, 0.5 * max(lengths) + 0.5 * sum(lengths) / len(lengths)))
        if re.search(r"comment|change|your answer|drives", rows[0][ci], re.I):
            w = max(w, 20)
        weights.append(w)
    if sum(minw) >= usable:
        return [usable * m / sum(minw) for m in minw]
    spare = usable - sum(minw)
    return [m + spare * w / sum(weights) for m, w in zip(minw, weights)]


def table(doc, lines):
    rows = [[c.strip() for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))] for ln in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    n = len(rows[0])
    rows = [(r + [""] * n)[:n] for r in rows]  # as many cells as the header row
    size, side = (8, 40) if n >= 10 else (9, 90)  # a wide grid (a domain-by-domain matrix): smaller type, less padding
    widths = col_widths(rows, size, side / 567)  # 567 twips per cm
    if n >= 10:  # a grid: a label column, then equal cells to write in
        first = min(max(widths[0], 1.6), 2.4)
        widths = [first] + [(PAGE_W - 2 * MARGIN - first) / (n - 1)] * (n - 1)
    t = doc.add_table(rows=len(rows), cols=n)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for ci, w in enumerate(widths):  # fixed layout follows the column grid, not the cell widths alone
        t.columns[ci].width = Cm(w)
    cell_margins(t, side=side)
    for ri, r in enumerate(rows):
        for ci, text in enumerate(r):
            cell = t.cell(ri, ci)
            cell.width = Cm(widths[ci])
            par = cell.paragraphs[0]
            par.paragraph_format.space_after = Pt(0)
            runs(par, text, size=size)
            if ri == 0:
                for run in par.runs:
                    run.bold = True
                shade(cell, "E7E6E6")
        if ri and not any(r):  # an empty row is space to write in
            min_height(t.rows[ri], 900 if n == 1 else 450)
    header = OxmlElement("w:tblHeader")  # repeat the header row on every page
    header.set(qn("w:val"), "true")
    t.rows[0]._tr.get_or_add_trPr().append(header)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def list_item(doc, kind, level, num, text):
    if kind == "b":
        par = doc.add_paragraph(style="List Bullet" if level == 0 else f"List Bullet {min(level + 1, 3)}")
        runs(par, text)
    else:
        par = doc.add_paragraph()
        par.paragraph_format.left_indent = Cm(0.75 + 0.6 * level)
        par.paragraph_format.first_line_indent = Cm(-0.5)
        runs(par, f"{num}. {text}")
    par.paragraph_format.space_after = Pt(2)


def note(doc, text):
    par = doc.add_paragraph()
    par.paragraph_format.left_indent = Cm(0.5)
    runs(par, text)
    for run in par.runs:
        run.italic = True
        run.font.color.rgb = RGBColor(0x40, 0x40, 0x40)


def code(doc, line):
    """One line of a code block, as written: monospace on a light shade, no inline markup."""
    par = doc.add_paragraph()
    shd = OxmlElement("w:shd")  # added before the spacing and indent, as the schema orders them
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), "F2F2F2")
    par._p.get_or_add_pPr().append(shd)
    par.paragraph_format.left_indent = Cm(0.3)
    par.paragraph_format.space_after = Pt(0)
    run = par.add_run(line)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)


def footer(section, text):
    par = section.footer.paragraphs[0]
    run = par.add_run(f"{text}  ·  page ")
    run.font.size = Pt(8)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    r, rpr, sz, t = OxmlElement("w:r"), OxmlElement("w:rPr"), OxmlElement("w:sz"), OxmlElement("w:t")
    sz.set(qn("w:val"), "16")
    t.text = "1"
    rpr.append(sz)
    r.append(rpr)
    r.append(t)
    field.append(r)
    par._p.append(field)


def convert(src, dst, footer_text=None):
    lines = Path(src).read_text(encoding="utf-8").splitlines()
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(MARGIN))
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)
    doc.styles["Normal"].paragraph_format.space_after = Pt(4)
    doc.styles["Title"].font.size = Pt(20)
    para, i = [], 0

    def flush():
        if para:
            runs(doc.add_paragraph(), " ".join(para))
            para.clear()

    while i < len(lines):
        line, s = lines[i], lines[i].strip()
        if not s or re.fullmatch(r"-{3,}", s):
            flush()
            i += 1
        elif s == "<!-- pagebreak -->":
            flush()
            doc.add_page_break()
            i += 1
        elif s.startswith("```"):  # a code block: each line as written, up to the closing fence
            flush()
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code(doc, lines[i])
                i += 1
            i += 1
            doc.add_paragraph().paragraph_format.space_after = Pt(0)
        elif s.startswith("☐"):  # answer options, one line each
            flush()
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Cm(0.6)
            par.paragraph_format.space_after = Pt(1)
            runs(par, s)
            i += 1
        elif s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            text = s[level:].strip()
            runs(doc.add_paragraph(style="Title" if level == 1 else f"Heading {min(level - 1, 3)}"), text)
            if level == 1 and not doc.core_properties.title:
                doc.core_properties.title = plain(text)
            i += 1
        elif s.startswith(">"):
            flush()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            note(doc, " ".join(quote))
        elif s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            table(doc, rows)
        elif BULLET.match(line) or NUMBER.match(line):
            flush()
            items = []
            while i < len(lines):
                mb, mn = BULLET.match(lines[i]), NUMBER.match(lines[i])
                if mb:
                    items.append(["b", len(mb.group(1)) // 2, None, mb.group(2)])
                elif mn:
                    items.append(["n", len(mn.group(1)) // 2, mn.group(2), mn.group(3)])
                elif lines[i].startswith("  ") and lines[i].strip() and items:
                    items[-1][3] += " " + lines[i].strip()
                else:
                    break
                i += 1
            for item in items:
                list_item(doc, *item)
        else:
            para.append(s)
            i += 1
    flush()
    footer(sec, footer_text or doc.core_properties.title)
    zoom = doc.settings.element.find(qn("w:zoom"))
    if zoom is not None and zoom.get(qn("w:percent")) is None:
        zoom.set(qn("w:percent"), "100")  # required by the schema; python-docx's template omits it
    doc.core_properties.author = ""
    doc.save(dst)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
