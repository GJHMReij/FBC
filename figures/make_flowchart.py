"""Writes flowchart_selection.svg (patient/scan selection). Edit the numbers in N below and re-run."""
N = {
    "all": "8,258", "assessable": "8,250 (1,438 PE)", "nonass": "8", "outpat": "568",
    "inpat": "7,682", "missing": "[ ]", "pop": "[ ]",
    "m3_nodd": "[ ]", "m3_pop": "[ ]",
    "tr12": "[ ]", "tr12_pe": "[ ]", "te12": "[ ]", "te12_pe": "[ ]",
    "tr3": "[ ]", "tr3_pe": "[ ]", "te3": "[ ]", "te3_pe": "[ ]",
}
W, H = 800, 860
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
box(110, 20, 360, 56, ["CTPA scans in the cohort", f"n = {N['all']}"])
line(cx, 76, cx, 130, True)
box(520, 75, 260, 56, ["Excluded", f"{N['nonass']} non-assessable CTPA"], RED)
line(cx, 103, 520, 103, True)
box(110, 130, 360, 56, ["Scans with an assessable CTPA", f"n = {N['assessable']}"])
line(cx, 186, cx, 240, True)
box(520, 185, 260, 56, ["Excluded", f"{N['outpat']} outpatient scans"], RED)
line(cx, 213, 520, 213, True)
box(110, 240, 360, 56, ["Inpatient and emergency scans", f"n = {N['inpat']}"])
line(cx, 296, cx, 350, True)
box(520, 295, 260, 56, ["Excluded", f"missing date, sex or age (n = {N['missing']})"], RED)
line(cx, 323, 520, 323, True)
box(110, 350, 360, 56, ["Analysis population", f"n = {N['pop']}"])
line(cx, 406, cx, 440)
line(200, 440, 600, 440)
line(200, 440, 200, 480, True)
line(600, 440, 600, 480, True)
box(20, 480, 360, 80, ["Models 1 and 2", "all scans in the analysis population", f"n = {N['pop']}"])
box(420, 480, 360, 80, ["Model 3", f"excluded: no measured D-dimer (n = {N['m3_nodd']})",
                        f"n = {N['m3_pop']}"])
for cxx, a, b, tr, te in [(200, 105, 295, "12", "12"), (600, 505, 695, "3", "3")]:
    line(cxx, 560, cxx, 595)
    line(a, 595, b, 595)
    line(a, 595, a, 630, True)
    line(b, 595, b, 630, True)
    box(a - 85, 630, 170, 90, ["Training set", f"n = {N['tr' + tr]} (80%)", f"{N['tr' + tr + '_pe']} PE"],
        fill=DARK)
    box(b - 85, 630, 170, 90, ["Hold-out test set", f"n = {N['te' + te]} (20%)", f"{N['te' + te + '_pe']} PE"],
        fill=GREEN)
out.append(f'<text x="{W / 2}" y="760" text-anchor="middle" font-size="12" fill="#2F5D45">'
           'Patients with scans in both periods are allocated entirely to the training set. '
           'The split is made after the exclusions.</text>')
out.append("</g></svg>")
open("figures/flowchart_selection.svg", "w", encoding="utf-8").write("\n".join(out))
print("written figures/flowchart_selection.svg")
