"""
Minimal Word report writer shared by try_models.py and univariable_auc.py.

write_report(path_base, title, paragraphs, sections): writes <path_base>.docx (python-docx) or, if
python-docx is not installed, <path_base>.html (Word opens it too). Returns the path written.
sections = [(heading, columns, rows, note)], rows = list of lists of strings. Aggregates only.
"""
from html import escape


def write_report(path_base, title, paragraphs, sections):
    path_base = str(path_base)
    try:
        import docx
        from docx.shared import Pt
    except ImportError:
        return _html(path_base, title, paragraphs, sections)

    doc = docx.Document()
    doc.add_heading(title, level=0)
    for p in paragraphs:
        doc.add_paragraph(p)
    for heading, cols, rows, note in sections:
        doc.add_heading(heading, level=1)
        if note:
            doc.add_paragraph(note)
        if not cols:
            continue
        t = doc.add_table(rows=1, cols=len(cols))
        t.style = "Light Grid Accent 1"
        for i, c in enumerate(cols):
            t.rows[0].cells[i].text = str(c)
        for r in rows:
            cells = t.add_row().cells
            for i, v in enumerate(r):
                cells[i].text = str(v)
        for row in t.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(9)
    path = path_base + ".docx"
    doc.save(path)
    return path


def _html(path_base, title, paragraphs, sections):
    out = [f"<html><head><meta charset='utf-8'><title>{escape(title)}</title></head><body>",
           f"<h1>{escape(title)}</h1>"]
    out += [f"<p>{escape(p)}</p>" for p in paragraphs]
    for heading, cols, rows, note in sections:
        out.append(f"<h2>{escape(heading)}</h2>")
        if note:
            out.append(f"<p>{escape(note)}</p>")
        if cols:
            out.append("<table border='1' cellspacing='0' cellpadding='3'><tr>"
                       + "".join(f"<th>{escape(str(c))}</th>" for c in cols) + "</tr>")
            for r in rows:
                out.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in r) + "</tr>")
            out.append("</table>")
    out.append("</body></html>")
    path = path_base + ".html"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    print("WARNING: python-docx not installed; wrote an HTML report instead")
    return path
