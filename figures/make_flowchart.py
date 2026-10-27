"""Writes flowchart_selection.svg (selection of scans and data split). Edit the numbers in N and re-run."""
import textwrap
from xml.sax.saxutils import escape

N = {
    "suspected": "8,202", "excluded_top": "512", "cohort": "7,690",
    "nonass": "6", "assessable": "7,684", "missing_date": "25", "missing_sexage": "1", "pop": "7,658",
    "m3_nodd": "2,910", "m3_pop": "4,748",
    "tr12": "6,126", "tr12_pe": "1,109", "te12": "1,532", "te12_pe": "270",
    "tr3": "3,798", "tr3_pe": "640", "te3": "950", "te3_pe": "164",
}


def pe_text(model, part):
    n = int(N[part + model].replace(",", ""))
    pe = int(N[part + model + "_pe"].replace(",", ""))
    return f"{N[part + model + '_pe']} PE ({100 * pe / n:.1f}%)"


W, H = 1080, 1050
GREEN, DARK, LIGHT, ARROW = "#54AA77", "#2F7D55", "#FFFFFF", "#2F5D45"
out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
       'font-family="Arial, Helvetica, sans-serif" font-size="14">',
       f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
       f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{ARROW}"/></marker></defs>',
       f'<rect width="{W}" height="{H}" fill="#fff"/>',
       f'<text x="{W / 2}" y="40" text-anchor="middle" font-size="28" font-weight="bold" fill="{GREEN}">'
       'Selection of scans and data split</text>',
       '<g transform="translate(0,60)">']


def box(x, y, w, h, lines, fill=GREEN):
    txt = "#FFFFFF" if fill != LIGHT else ARROW
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" '
               f'stroke="{DARK}" stroke-width="1.2"/>')
    y0 = y + h / 2 - (len(lines) - 1) * 9 + 5
    for i, t in enumerate(lines):
        wt = ' font-weight="bold"' if i == 0 else ""
        out.append(f'<text x="{x + w / 2}" y="{y0 + i * 18}" text-anchor="middle" fill="{txt}"{wt}>{escape(t)}</text>')


def line(x1, y1, x2, y2, arrow=False):
    m = ' marker-end="url(#a)"' if arrow else ""
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{ARROW}" stroke-width="1.5"{m}/>')


cx, BH, STEP = 400, 52, 100
rows = [
    (["CT scans performed due to suspected PE", f"n = {N['suspected']}"], None),
    (["CTPA scans in the cohort", f"n = {N['cohort']}"],
     ["Excluded", f"scans < 1 month after previous scan (n = {N['excluded_top']})"]),
    (["Scans with an assessable CTPA", f"n = {N['assessable']}"],
     ["Excluded", f"CTPA report missing or not assessable (n = {N['nonass']})"]),
    (["Analysis population", f"n = {N['pop']}"],
     ["Excluded", f"missing scan date (n = {N['missing_date']})", f"missing sex or age (n = {N['missing_sexage']})"]),
]
for i, (main, excl) in enumerate(rows):
    y = 20 + i * STEP
    box(cx - 200, y, 400, BH, main)
    if i > 0:
        line(cx, y - STEP + BH, cx, y, True)
        mid = y - (STEP - BH) / 2
        wrapped = [excl[0]] + [w for t in excl[1:] for w in textwrap.wrap(t, 34)]
        h = max(56, 18 * len(wrapped) + 20)
        box(630, mid - h / 2, 270, h, wrapped, LIGHT)
        line(cx, mid, 630, mid, True)

last = 20 + (len(rows) - 1) * STEP + BH          # bottom of the analysis population box
split_y, row5 = last + 38, last + 98
line(cx, last, cx, split_y)
line(200, split_y, 600, split_y)
line(200, split_y, 200, row5, True)
line(600, split_y, 600, row5, True)
mid3 = split_y + (row5 - split_y) / 2
box(800, mid3 - 28, 260, 56, ["Excluded", f"D-dimer missing (n = {N['m3_nodd']}; {100 * int(N['m3_nodd'].replace(',', '')) / int(N['pop'].replace(',', '')):.1f}%)"], LIGHT)
line(600, mid3, 800, mid3, True)
box(20, row5, 360, 80, ["Models 1 and 2", "all scans in the analysis population", f"n = {N['pop']}"])
box(420, row5, 360, 80, ["Model 3", "scans with a measured D-dimer", f"n = {N['m3_pop']}"])
b0 = row5 + 80
sp, row6 = b0 + 35, b0 + 70
for cxx, a, b, tr in [(200, 105, 295, "12"), (600, 505, 695, "3")]:
    line(cxx, b0, cxx, sp)
    line(a, sp, b, sp)
    line(a, sp, a, row6, True)
    line(b, sp, b, row6, True)
    box(a - 85, row6, 170, 90, ["Training set", f"n = {N['tr' + tr]} (80%)", pe_text(tr, "tr")], DARK)
    box(b - 85, row6, 170, 90, ["Hold-out test set", f"n = {N['te' + tr]} (20%)", pe_text(tr, "te")], GREEN)
out.append(f'<text x="{W / 2}" y="{row6 + 125}" text-anchor="middle" font-size="12" fill="{ARROW}">'
           'Patients with scans in both periods (n = 72 for Models 1 and 2; n = 45 for Model 3) '
           'are allocated entirely to the training set.</text>')
out.append("</g></svg>")
open("figures/flowchart_selection.svg", "w", encoding="utf-8").write("\n".join(out))
print("written figures/flowchart_selection.svg")
