"""Writes flowchart_selection.svg (patient/scan selection). Edit the numbers in N below and re-run."""
N = {
    "cohort": "7,690", "nonass": "6", "assessable": "7,684", "missing": "26", "pop": "7,658",
    "m3_nodd": "2,910", "m3_pop": "4,748",
    "tr12": "6,126", "tr12_pe": "1,109", "te12": "1,532", "te12_pe": "270",
    "tr3": "3,798", "tr3_pe": "640", "te3": "950", "te3_pe": "164",
}
W, H = 800, 740
out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
       'font-family="Arial, Helvetica, sans-serif" font-size="14">',
       '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
       'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#2F5D45"/></marker></defs>',
       f'<rect width="{W}" height="{H}" fill="#fff"/>',
       f'<text x="{W / 2}" y="40" text-anchor="middle" font-size="28" font-weight="bold" fill="#54AA77">Selection of scans and data split</text>',
       '<g transform="translate(0,60)">']


GREEN, DARK, LIGHT = "#54AA77", "#2F7D55", "#FFFFFF"


def box(x, y, w, h, lines, fill=GREEN):
    txt = "#FFFFFF" if fill != LIGHT else "#2F5D45"
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" '
               f'stroke="{DARK}" stroke-width="1.2"/>')
    y0 = y + h / 2 - (len(lines) - 1) * 9 + 5
    for i, t in enumerate(lines):
        wt = ' font-weight="bold"' if i == 0 else ""
        out.append(f'<text x="{x + w / 2}" y="{y0 + i * 18}" text-anchor="middle" fill="{txt}"{wt}>{t}</text>')


def line(x1, y1, x2, y2, arrow=False):
    m = ' marker-end="url(#a)"' if arrow else ""
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#2F5D45" stroke-width="1.5"{m}/>')


RED = LIGHT
cx = 290
box(110, 20, 360, 56, ["CTPA scans in the cohort", f"n = {N['cohort']}"])
line(cx, 76, cx, 130, True)
box(520, 75, 260, 56, ["Excluded", f"{N['nonass']} non-assessable CTPA"], RED)
line(cx, 103, 520, 103, True)
box(110, 130, 360, 56, ["Scans with an assessable CTPA", f"n = {N['assessable']}"])
line(cx, 186, cx, 240, True)
box(520, 185, 260, 56, ["Excluded", f"missing date, sex or age (n = {N['missing']})"], RED)
line(cx, 213, 520, 213, True)
box(110, 240, 360, 56, ["Analysis population", f"n = {N['pop']}"])
line(cx, 296, cx, 330)
line(200, 330, 600, 330)
line(200, 330, 200, 370, True)
line(600, 330, 600, 370, True)
box(20, 370, 360, 80, ["Models 1 and 2", "all scans in the analysis population", f"n = {N['pop']}"])
box(420, 370, 360, 80, ["Model 3", f"excluded: no measured D-dimer (n = {N['m3_nodd']})",
                        f"n = {N['m3_pop']}"])
for cxx, a, b, tr, te in [(200, 105, 295, "12", "12"), (600, 505, 695, "3", "3")]:
    line(cxx, 450, cxx, 485)
    line(a, 485, b, 485)
    line(a, 485, a, 520, True)
    line(b, 485, b, 520, True)
    box(a - 85, 520, 170, 90, ["Training set", f"n = {N['tr' + tr]} (80%)", f"{N['tr' + tr + '_pe']} PE"],
        fill=DARK)
    box(b - 85, 520, 170, 90, ["Hold-out test set", f"n = {N['te' + te]} (20%)", f"{N['te' + te + '_pe']} PE"],
        fill=GREEN)
out.append(f'<text x="{W / 2}" y="650" text-anchor="middle" font-size="12" fill="#2F5D45">'
           'Patients with scans in both periods (n = 72 for Models 1 and 2; n = 45 for Model 3) '
           'are allocated entirely to the training set.</text>')
out.append("</g></svg>")
open("figures/flowchart_selection.svg", "w", encoding="utf-8").write("\n".join(out))
print("written figures/flowchart_selection.svg")
