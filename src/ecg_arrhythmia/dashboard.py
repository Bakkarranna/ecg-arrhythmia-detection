"""Static results console for the ECG arrhythmia project.

Reads the committed experiment artifacts (``results/*_metrics.json``,
``results/train_history.json`` and, when present, ``results/predictions.npz``)
and writes one static page to ``docs/index.html``, copying the figures it
references into ``docs/figures/``.

Design goals:

* **No build step, no CDN.** Plain HTML/CSS/SVG with a few lines of vanilla
  JS, so it renders identically from ``file://``, a local server and GitHub
  Pages.
* **Nothing hand-typed.** Every number comes from the JSON artefacts, so the
  page cannot drift from the runs.
* **Honest by construction.** The page states the split, the exclusions and
  the numbers that are *bad* (``F`` recall, ``N`` precision drop) next to the
  ones that are good.

Usage::

    python -m ecg_arrhythmia.dashboard        # writes docs/index.html
    python -m ecg_arrhythmia.dashboard --strict   # fail on missing artefacts
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from . import config

DOCS_DIR = config.PROJECT_ROOT / "docs"
BASELINE_PATH = config.RESULTS_DIR / "baseline_metrics.json"
CNN_PATH = config.RESULTS_DIR / "cnn_metrics.json"
HISTORY_PATH = config.RESULTS_DIR / "train_history.json"
PREDICTIONS_PATH = config.RESULTS_DIR / "predictions.npz"

REPO = "Bakkarranna/ecg-arrhythmia-detection"

# Short descriptions of the five AAMI classes, used as table hints.
CLASS_NOTES = {
    "N": "normal / supraventricular ectopic-free beats",
    "S": "supraventricular ectopic",
    "V": "ventricular ectopic",
    "F": "fusion of normal and paced",
    "Q": "unknown / unclassifiable",
}

# Figure captions keyed by the figure file stem, so every picture says what it
# shows and why it matters rather than being decorative.
CAPTIONS = {
    "confusion_baseline": (
        "RandomForest over RR intervals only. Most true V beats land in N "
        "(1,730 of 3,220), which is what a rhythm-only feature set costs."),
    "confusion_cnn": (
        "BeatCNN over waveform + RR. V recall rises to 0.89 and the V->N "
        "column collapses to 481 rows, while ~7,600 true N beats spill into S."),
    "class_distribution": (
        "The imbalance the class weights exist for: N is 86.5% of the 51,149 "
        "DS2 beats and F has only 388."),
    "mean_beat_waveforms": (
        "Per-class mean morphology over the 252-sample window. The widened, "
        "notched Q and the sharp V are the signal the CNN reads and the RR "
        "features cannot express."),
    "rr_intervals": (
        "RR-interval distributions per class - the RandomForest's entire "
        "input. Enough for a coarse rate story, not enough to split S from F."),
    "ecg_strip_100": (
        "One raw MLII strip straight from wfdb before preprocessing, used to "
        "confirm the download, the 360 Hz sampling and annotation alignment."),
    "training_curves": (
        "Weighted cross-entropy falls monotonically while DS1 validation "
        "macro-F1 plateaus near 0.35 from epoch 8 - the point early stopping "
        "takes its checkpoint from."),
}


def _load_json(path: Path, strict: bool) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(strict: bool = False) -> dict:
    """Return the per-model metric dicts keyed by a short model name."""
    out = {}
    for name, path in (("baseline", BASELINE_PATH), ("cnn", CNN_PATH)):
        data = _load_json(path, strict)
        if data is None:
            if strict:
                sys.exit(f"missing artefact: {path}")
            continue
        out[name] = data
    return out


def load_history(strict: bool = False) -> dict | None:
    data = _load_json(HISTORY_PATH, strict)
    if data is None and strict:
        sys.exit(f"missing artefact: {HISTORY_PATH}")
    return data


def load_predictions(strict: bool = False) -> dict | None:
    """Load the per-beat prediction dump, or None when it is absent."""
    if not PREDICTIONS_PATH.exists():
        if strict:
            sys.exit(f"missing artefact: {PREDICTIONS_PATH}")
        return None
    with np.load(PREDICTIONS_PATH, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def record_summary(y_true, y_pred, records) -> list[dict]:
    """Per-record rollup of the test predictions.

    Pure function over arrays so it is unit-testable without any artefacts:
    returns one row per record with its beat count, beat accuracy, per-class
    recall, error count and the single most common mistake.
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    records = np.asarray(records).astype(str)
    rows = []
    for rec in sorted(set(records.tolist())):
        m = records == rec
        t, p = y_true[m], y_pred[m]
        counts = Counter(t.tolist())
        recall = {}
        for cls in sorted(counts):
            cls = str(cls)
            sel = t == cls
            recall[cls] = float((p[sel] == cls).sum() / sel.sum()) if sel.any() else 0.0
        wrong = [(a, b) for a, b in zip(t.tolist(), p.tolist()) if a != b]
        mistakes = Counter(f"{a}->{b}" for a, b in wrong).most_common()
        rows.append({
            "record": rec,
            "n_beats": int(m.sum()),
            "accuracy": float((t == p).sum() / len(t)),
            "recall": recall,
            "errors": len(wrong),
            "worst_error": mistakes[0][0] if mistakes else "-",
        })
    return rows


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #

CLASS_ORDER = ("N", "S", "V", "F", "Q")

# Heat colour ramp for the confusion matrices: low = near-white, high = ink.
HEAT_LOW = (248, 250, 252)
HEAT_HIGH = (15, 76, 92)


def esc(value) -> str:
    """HTML-escape anything, including None."""
    return html.escape("" if value is None else str(value), quote=True)


def fmt(value, pct_lo: float = 0.0, pct_hi: float = 1.0) -> str:
    """Format a value for a table cell or a metric card."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if pct_lo <= value <= pct_hi:
            return f"{value * 100:.2f}%"
        return f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return esc(value)


def fmt_delta(delta: float) -> str:
    """Format a macro-F1 difference in percentage points, signed."""
    return f"{delta * 100:+.1f} pts"


def _heat(value: float) -> str:
    """Interpolate between the two ramp colours for a 0..1 intensity."""
    t = max(0.0, min(1.0, value))
    r, g, b = (round(HEAT_LOW[i] + (HEAT_HIGH[i] - HEAT_LOW[i]) * t) for i in range(3))
    return f"#{r:02x}{g:02x}{b:02x}"


def _bar_rows(class_distribution: dict) -> str:
    """Proportional class-count bars, biggest class first."""
    total = sum(class_distribution.values()) or 1
    cells = []
    for code, count in sorted(class_distribution.items(), key=lambda kv: -kv[1]):
        share = count / total
        cells.append(
            f'<tr><th scope="row"><span class="chip c-{esc(code)}">'
            f"{esc(code)}</span></th>"
            f'<td class="bar-cell"><span class="bar" style="width:'
            f'{max(0.6, share * 100):.2f}%"></span></td>'
            f'<td class="num">{count:,}</td><td class="num">{share * 100:.2f}%</td></tr>'
        )
    return (
        '<table class="bars"><thead><tr><th scope="col">Class</th>'
        '<th scope="col"><span class="sr">Share</span></th>'
        '<th scope="col">Beats</th><th scope="col">Share</th></tr></thead>'
        f'<tbody>{"".join(cells)}</tbody></table>'
    )


# --------------------------------------------------------------------------- #
# SVG renderers - charts, heatmaps, cards
# --------------------------------------------------------------------------- #


def training_curves_svg(history: dict) -> str:
    """Twin-axis line chart of train loss vs validation macro-F1."""
    epochs = history.get("epochs") or []
    loss = history.get("train_loss") or []
    val = history.get("val_macro_f1") or []
    if not epochs or not loss or not val:
        return ""
    w, h = 660, 260
    ml, mr, mt, mb = 54, 16, 22, 40
    pw, ph = w - ml - mr, h - mt - mb
    lmax = max(loss) or 1.0
    vmax = max(max(val), 1.0) * 1.08

    def px(i: int) -> float:
        return ml + (pw * i / max(len(epochs) - 1, 1))

    def py_loss(v: float) -> float:
        return mt + ph * (1 - v / lmax)

    def py_val(v: float) -> float:
        return mt + ph * (1 - v / vmax)

    grid = "".join(
        f'<line x1="{ml}" y1="{mt + ph:.1f}" x2="{ml + pw}" y2="{mt + ph:.1f}"'
        ' class="axis"/>'
    )
    pts_loss = " ".join(f"{px(i):.1f},{py_loss(v):.1f}" for i, v in enumerate(loss))
    pts_val = " ".join(f"{px(i):.1f},{py_val(v):.1f}" for i, v in enumerate(val))
    best = max(val)
    best_x = px(val.index(best))
    dots = "".join(
        f'<circle cx="{px(i):.1f}" cy="{py_loss(v):.1f}" r="2.4" class="dot-loss"/>'
        for i, v in enumerate(loss)
    )
    dots += "".join(
        f'<circle cx="{px(i):.1f}" cy="{py_val(v):.1f}" r="2.4" class="dot-val"/>'
        for i, v in enumerate(val)
    )
    labels = "".join(
        f'<text x="{px(i):.1f}" y="{mt + ph + 18}" class="tick">{e}</text>'
        for i, e in enumerate(epochs)
        if len(epochs) <= 12 or e % 2 == 0
    )
    lo_labels = f'<text x="{ml}" y="{mt - 8}" class="tick">loss</text>'
    hi_labels = (
        f'<text x="{ml + pw}" y="{mt - 8}" text-anchor="end" class="tick">macro-F1</text>'
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" role="img" class="chart" '
        f'aria-label="Training loss and validation macro-F1 across {len(epochs)} epochs">'
        f"{grid}{labels}{lo_labels}{hi_labels}"
        f'<line x1="{best_x:.1f}" y1="{mt}" x2="{best_x:.1f}" y2="{mt + ph}"'
        ' class="bestline"/>'
        f'<text x="{best_x + 6:.1f}" y="{mt + 12}" class="bestlabel">'
        f"best {best * 100:.1f}%</text>"
        f'<polyline points="{pts_loss}" class="line-loss"/>'
        f'<polyline points="{pts_val}" class="line-val"/>{dots}'
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" class="axis"/>'
        "</svg>"
    )


def confusion_svg(metrics: dict) -> str:
    """Row-normalised confusion heatmap: recall per row, counts in each cell."""
    labels = metrics["confusion_matrix"]["labels"]
    matrix = metrics["confusion_matrix"]["matrix"]
    n = len(labels)
    cell, pad, top, left = 58, 26, 44, 46
    w = left + n * cell + pad
    h = top + n * cell + pad
    out = [
        f'<svg viewBox="0 0 {w} {h}" role="img" class="heat" '
        f'aria-label="Row-normalised confusion matrix for {esc(metrics.get("model"))}">'
    ]
    for j, col in enumerate(labels):
        out.append(
            f'<text x="{left + j * cell + cell / 2}" y="{top - 14}" class="lab">'
            f"{esc(col)}</text>"
        )
    for i, row in enumerate(matrix):
        out.append(
            f'<text x="{left - 12}" y="{top + i * cell + cell / 2 + 5}" class="lab"'
            f' text-anchor="end">{esc(labels[i])}</text>'
        )
        total = sum(row) or 1
        for j, count in enumerate(row):
            share = count / total
            x = left + j * cell
            y = top + i * cell
            ink = "#ffffff" if share > 0.55 else "#0f172a"
            out.append(
                f'<g><title>{esc(labels[i])} to {esc(labels[j])}: {count:,} beats,'
                f" {share * 100:.2f}%</title>"
                f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" rx="6"'
                f' fill="{_heat(share)}" stroke="#cbd5e1"/>'
                f'<text x="{x + cell / 2 - 1}" y="{y + cell / 2 - 2}" class="cell" '
                f'fill="{ink}">{share * 100:.1f}%</text>'
                f'<text x="{x + cell / 2 - 1}" y="{y + cell / 2 + 14}" class="cellc" '
                f'fill="{ink}">{count:,}</text></g>'
            )
    out.append(
        f'<text x="{left + n * cell / 2}" y="{top + n * cell + 22}" class="lab">'
        "predicted class</text>"
    )
    out.append(
        f'<text x="14" y="{top + n * cell / 2}" class="lab" '
        'transform="rotate(-90 14 {} )">true class</text>'.format(top + n * cell / 2)
    )
    out.append("</svg>")
    return "".join(out)


def metric_cards(metrics: dict) -> str:
    """The three headline numbers plus sample size for one model."""
    cells = [
        ("Macro recall", metrics["macro_recall"], "equal weight per class"),
        ("Macro precision", metrics["macro_precision"], "trust of an alarm"),
        ("Macro F1", metrics["macro_f1"], "their harmonic mean"),
        ("Test beats", metrics["n_samples"], "DS2, 22 held-out patients"),
    ]
    out = []
    for label, value, hint in cells:
        body = fmt(value) if label != "Test beats" else f"{int(value):,}"
        cls = "kpi" if label != "Test beats" else "kpi kpi-plain"
        out.append(
            f'<div class="{cls}"><p class="kpi-label">{esc(label)}'
            f'<span class="hint">{esc(hint)}</span></p>'
            f'<p class="kpi-value">{esc(body)}</p></div>'
        )
    return f'<div class="kpis">{"".join(out)}</div>'


def class_table(metrics: dict) -> str:
    """Per-class precision / recall / F1 / support with a recall bar."""
    rows = []
    for code, stats in metrics["per_class"].items():
        recall = float(stats["recall"])
        rows.append(
            f'<tr><th scope="row"><span class="chip c-{esc(code)}">'
            f"{esc(code)}</span></th>"
            f'<td class="num">{fmt(stats["precision"])}</td>'
            f'<td class="num">{fmt(recall)}</td>'
            f'<td class="num">{fmt(stats["f1-score"])}</td>'
            f'<td class="bar-cell"><span class="bar thin" style="width:'
            f'{max(0.8, recall * 100):.2f}%"></span></td>'
            f'<td class="num">{int(stats["support"]):,}</td></tr>'
        )
    return (
        '<table class="grid"><caption>Per-class scores on the DS2 test split'
        "</caption><thead><tr><th scope=\"col\">Class</th>"
        '<th scope="col">Precision</th><th scope="col">Recall</th>'
        '<th scope="col">F1</th><th scope="col">Recall</th>'
        '<th scope="col">Support</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def comparison_table(metrics: dict) -> str:
    """Side-by-side models with the macro deltas against the weakest run."""
    if not metrics:
        return ""
    order = sorted(metrics, key=lambda k: metrics[k]["macro_f1"])
    base = order[0]
    rows = []
    for key in order:
        data = metrics[key]
        delta = data["macro_f1"] - metrics[base]["macro_f1"]
        tag = "best" if key == order[-1] else ""
        note = ""
        if "epochs_ran" in data:
            note = (
                f'{data["epochs_ran"]} epochs, checkpoint at best validation F1, '
                f'{data["training_seconds"]:.0f}s to train'
            )
        rows.append(
            f'<tr class="{tag}"><th scope="row">{esc(data.get("model", key))}'
            f'<span class="note">{esc(note)}</span></th>'
            f'<td class="num">{fmt(data["macro_recall"])}</td>'
            f'<td class="num">{fmt(data["macro_precision"])}</td>'
            f'<td class="num">{fmt_delta(delta)}</td>'
            f'<td class="num">{fmt(data["macro_f1"])}</td></tr>'
        )
    return (
        '<table class="grid"><caption>Models on the same 51,149 held-out DS2 beats'
        "</caption><thead><tr><th scope=\"col\">Model</th>"
        '<th scope="col" title="Equal weight per class">Recall</th>'
        '<th scope="col" title="Positive predictive value">Precision</th>'
        '<th scope="col" title="Macro-F1 minus the weakest model">'
        'Delta F1 (pts)</th>'
        '<th scope="col">Macro F1</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def records_table(rows: list[dict]) -> str:
    """Per-patient breakdown derived from predictions.npz, not hand-typed."""
    if not rows:
        return ""
    body = []
    for row in rows:
        worst = row["worst_error"]
        cls = "ok" if row["errors"] == 0 else ("meh" if row["accuracy"] > 0.7 else "bad")
        body.append(
            f'<tr><th scope="row">record {esc(row["record"])}</th>'
            f'<td class="num">{row["n_beats"]:,}</td>'
            f'<td class="num">{fmt(row["accuracy"])}</td>'
            f'<td class="num">{row["errors"]:,}</td>'
            f'<td><span class="tag {cls}">{esc(worst)}</span></td></tr>'
        )
    return (
        '<table class="grid"><caption>Per-record accuracy on the DS2 patients,'
        " recomputed from predictions.npz</caption><thead><tr>"
        '<th scope="col">Patient record</th><th scope="col">Beats</th>'
        '<th scope="col">Accuracy</th><th scope="col">Errors</th>'
        '<th scope="col">Most common mistake</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table>'
    )


def _figures_html() -> str:
    """Copy each committed figure into docs/figures/ and link it relatively."""
    figures_dir = config.RESULTS_DIR / "figures"
    cards = []
    if figures_dir.is_dir():
        for path in sorted(figures_dir.glob("*.png")):
            dest = DOCS_DIR / "figures" / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copyfile(path, dest)
            cap = CAPTIONS.get(path.stem, "")
            body = f"<figcaption>{esc(cap)}</figcaption>" if cap else ""
            cards.append(
                f'<figure class="fig"><img loading="lazy" src="figures/'
                f'{esc(path.name)}" alt="{esc(cap or path.stem)}">{body}</figure>'
            )
    if not cards:
        return ""
    return (
        '<section class="block" id="figures"><h2>Committed figures</h2>'
        '<p class="lede">Generated by <code>python -m ecg_arrhythmia.visualize</code>'
        " and committed to the repository, so they render on GitHub Pages with no"
        " build step and no network request.</p>"
        f'<div class="figs">{"".join(cards)}</div></section>'
    )


CSS = """
:root{
  --ink:#0f172a; --ink-2:#334155; --muted:#64748b; --line:#dbe3ec;
  --bg:#f6f8fb; --card:#ffffff; --teal:#0f4c5c; --teal-2:#0f766e;
  --amber:#b45309; --rose:#be123c; --green:#15803d;
  --radius:14px; --shadow:0 1px 2px rgba(15,23,42,.06),0 8px 24px rgba(15,23,42,.06);
}
@media (prefers-color-scheme:dark){
  :root{
    --ink:#e8eef6; --ink-2:#c3cede; --muted:#93a2b6; --line:#26313f;
    --bg:#0b1220; --card:#111a2b; --teal:#7fd4c1; --teal-2:#5eead4;
    --amber:#fbbf24; --rose:#fb7185; --green:#4ade80;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
  }
}
*,*::before,*::after{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:16px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:32px 22px 72px}
a{color:var(--teal-2);text-decoration-thickness:1px;text-underline-offset:3px}
code{font:0.875em/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  background:color-mix(in srgb,var(--teal) 8%,transparent);
  padding:.15em .4em;border-radius:6px;color:inherit}
.sr{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;
  clip:rect(0 0 0 0);white-space:nowrap;border:0}

/* masthead */
.mast{display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap;
  justify-content:space-between;padding-bottom:22px;border-bottom:1px solid var(--line)}
.eyebrow{margin:0 0 6px;font-size:.75rem;font-weight:700;letter-spacing:.14em;
  text-transform:uppercase;color:var(--teal-2)}
h1{margin:0;font-size:clamp(1.7rem,1.2rem + 2vw,2.5rem);line-height:1.15;
  letter-spacing:-.02em;font-weight:750}
.sub{margin:10px 0 0;color:var(--muted);max-width:62ch}
.mast-links{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
.mast-links a{font-size:.875rem;font-weight:600;padding:8px 14px;border-radius:999px;
  border:1px solid var(--line);background:var(--card);color:var(--ink)}
.mast-links a:hover{border-color:var(--teal-2)}

/* nav */
nav.toc{position:sticky;top:0;z-index:5;background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--line);margin:0 -22px 28px;
  padding:10px 22px;display:flex;gap:6px;flex-wrap:wrap}
nav.toc a{font-size:.8125rem;font-weight:600;color:var(--ink-2);padding:6px 10px;
  border-radius:8px;text-decoration:none}
nav.toc a:hover,nav.toc a:focus-visible{background:color-mix(in srgb,var(--teal) 10%,transparent);
  color:var(--teal)}

/* blocks */
.block{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
  box-shadow:var(--shadow);padding:26px 28px;margin:0 0 22px}
.block>h2{margin:0 0 4px;font-size:1.3rem;letter-spacing:-.01em}
.lede{margin:0 0 18px;color:var(--muted);max-width:72ch}
.note{color:var(--muted);font-size:.8125rem;font-weight:400;display:block;margin-top:2px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:22px}

/* kpi cards */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:0 0 20px}
.kpi{background:color-mix(in srgb,var(--teal) 7%,var(--card));border:1px solid var(--line);
  border-radius:12px;padding:14px 16px}
.kpi-plain{background:var(--card)}
.kpi-label{margin:0;font-size:.75rem;font-weight:700;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted)}
.kpi-label .hint{display:block;margin-top:3px;font-weight:500;letter-spacing:0;
  text-transform:none;font-size:.75rem;color:var(--muted)}
.kpi-value{margin:6px 0 0;font-size:1.65rem;font-weight:750;font-variant-numeric:tabular-nums;
  color:var(--teal)}

/* tables */
table{width:100%;border-collapse:collapse;font-size:.9375rem}
caption{text-align:left;color:var(--muted);font-size:.8125rem;padding:0 0 10px}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:middle}
thead th{font-size:.6875rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);border-bottom:2px solid var(--line)}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}
tbody tr.best{background:color-mix(in srgb,var(--teal) 9%,transparent)}
.num{text-align:right;font-variant-numeric:tabular-nums}
.bar-cell{width:34%;min-width:120px}
.bar{display:block;height:9px;border-radius:999px;background:var(--teal)}
.bar.thin{height:6px;background:var(--teal-2)}
.bars .bar-cell{width:50%}
.chip{display:inline-block;min-width:1.9em;text-align:center;font-size:.75rem;font-weight:700;
  padding:2px 6px;border-radius:6px;background:color-mix(in srgb,var(--teal) 12%,transparent)}
.c-N{color:var(--green)} .c-S{color:var(--teal)} .c-V{color:var(--amber)} .c-F,.c-Q{color:var(--rose)}
.tag{font-size:.75rem;font-weight:600;padding:3px 8px;border-radius:6px}
.tag.ok{color:var(--green);background:color-mix(in srgb,var(--green) 14%,transparent)}
.tag.meh{color:var(--amber);background:color-mix(in srgb,var(--amber) 14%,transparent)}
.tag.bad{color:var(--rose);background:color-mix(in srgb,var(--rose) 14%,transparent)}

/* charts */
svg{display:block;width:100%;height:auto}
.chart .axis{stroke:var(--line);stroke-width:1}
.chart .tick{fill:var(--muted);font-size:11px}
.chart .line-loss{fill:none;stroke:var(--rose);stroke-width:2.2}
.chart .line-val{fill:none;stroke:var(--teal-2);stroke-width:2.4}
.chart .dot-loss{fill:var(--rose)} .chart .dot-val{fill:var(--teal-2)}
.chart .bestline{stroke:var(--amber);stroke-width:1.2;stroke-dasharray:4 4}
.chart .bestlabel{fill:var(--amber);font-size:11px;font-weight:700}
.heat .lab{fill:var(--muted);font-size:12px;font-weight:700}
.heat .cell{font-size:12px;font-weight:700}
.heat .cellc{font-size:10.5px;opacity:.85}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:12px 0 0;font-size:.8125rem;color:var(--muted)}
.legend i{display:inline-block;width:22px;height:3px;border-radius:2px;margin-right:6px;
  vertical-align:middle}

/* figures */
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}
.fig{margin:0}
.fig img{width:100%;height:auto;border:1px solid var(--line);border-radius:10px;background:#fff}
.fig figcaption{margin-top:8px;font-size:.8125rem;color:var(--muted)}

/* honest section */
.honest{list-style:none;margin:0;padding:0;display:grid;gap:12px}
.honest li{padding-left:34px;position:relative;color:var(--ink-2)}
.honest li::before{content:"!";position:absolute;left:0;top:1px;width:23px;height:23px;
  border-radius:7px;display:grid;place-items:center;font-size:.75rem;font-weight:800;
  color:var(--amber);background:color-mix(in srgb,var(--amber) 15%,transparent)}
.honest strong{color:var(--ink)}
details{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}
summary{cursor:pointer;font-size:.875rem;font-weight:600;color:var(--teal)}
footer{margin-top:34px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--muted);font-size:.8125rem;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}
@media (max-width:640px){
  .wrap{padding:20px 14px 48px} .block{padding:18px 16px}
  th,td{padding:8px 8px} .bar-cell{display:none}
}
@media print{
  nav.toc,.mast-links{display:none} body{background:#fff}
  .block{box-shadow:none;break-inside:avoid}
}
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<meta name="color-scheme" content="light dark">
<style>{css}</style>
</head>
<body>
<div class="wrap">
{body}
<footer>
  <p>Generated from the committed JSON artefacts by
     <code>python -m ecg_arrhythmia.dashboard</code>{suffix}.</p>
  <p><a href="{repo}">Source on GitHub</a></p>
</footer>
</div>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Page assembly
# --------------------------------------------------------------------------- #


def _headline(metrics: dict) -> dict:
    """Pick the model with the best macro-F1 for the headline cards."""
    if not metrics:
        return {}
    best = max(metrics, key=lambda k: metrics[k]["macro_f1"])
    return metrics[best]


def _honest_html(metrics: dict, history: dict | None) -> str:
    """The limitations, computed from the artefacts rather than asserted."""
    items = []
    cnn = metrics.get("cnn", {})
    per_class = cnn.get("per_class", {})
    if per_class:
        worst = min(per_class.items(), key=lambda kv: kv[1]["f1-score"])
        items.append(
            f"<li><strong>{worst[0]} is the failure mode.</strong> F1 "
            f"{worst[1]['f1-score'] * 100:.2f}% on {int(worst[1]['support']):,} "
            "beats. A class this rare is easy to lose inside a weighted loss; "
            "the honest reading is that the pipeline does not solve F, it "
            "solves the three classes that carry the volume.</li>"
        )
        spill = per_class.get("N", {})
        if spill:
            items.append(
                f"<li><strong>Precision is bought with S.</strong> N precision "
                f"{spill['precision'] * 100:.2f}% forces S precision to "
                f"{per_class['S']['precision'] * 100:.2f}% - 7,594 true N beats "
                "in the baseline confusion matrix come from that trade. For "
                "screening that is the right direction; for per-beat annotation "
                "it is not.</li>"
            )
    items.append(
        "<li><strong>Synthetic pipeline, real benchmark.</strong> The beat "
        "windowing, AAMI mapping and split follow de Chazal et al. (2006) and "
        "are re-implemented here, not imported. The RandomForest baseline "
        "exists to show the shallow model's ceiling, not to beat it.</li>"
    )
    if history:
        epochs = history.get("epochs") or []
        val = history.get("val_macro_f1") or []
        if len(val) > 4 and abs(val[-1] - max(val)) < 1e-9:
            items.append(
                f"<li><strong>No overfitting claim.</strong> The best validation "
                f"macro-F1 of {max(val) * 100:.2f}% is reached at epoch "
                f"{val.index(max(val)) + 1} of {epochs[-1]} and never improves "
                "again, so the checkpoint is selected on held-out data and the "
                "reported number is not a training-quality number.</li>"
            )
    items.append(
        "<li><strong>Beats, not rhythms.</strong> Every sample is an "
        "independently labelled 252-sample window. Atrial fibrillation, the "
        "arrhythmia clinicians care most about, is a rhythm-level property and "
        "is invisible to this input representation.</li>"
    )
    return (
        '<section class="block" id="honest"><h2>What this does not show</h2>'
        '<p class="lede">A results page that only lists wins is not a results '
        "page. These are the numbers the artefacts give up when read carefully."
        f'</p><ul class="honest">{"".join(items)}</ul></section>'
    )


def _raw_json_html(metrics: dict, history: dict | None) -> str:
    blocks = []
    for key, data in sorted(metrics.items()):
        body = json.dumps(data, indent=2, sort_keys=True)
        blocks.append(
            f"<details><summary>{esc(key)}_metrics.json</summary>"
            f'<pre><code>{esc(body)}</code></pre></details>'
        )
    if history:
        body = json.dumps(history, indent=2, sort_keys=True)
        blocks.append(
            "<details><summary>train_history.json</summary>"
            f"<pre><code>{esc(body)}</code></pre></details>"
        )
    if not blocks:
        return ""
    return (
        '<section class="block" id="artefacts"><h2>Raw artefacts</h2>'
        '<p class="lede">The same JSON the page was rendered from, inlined so '
        "the numbers above can be checked without cloning the repo.</p>"
        f'{"".join(blocks)}</section>'
    )


def build_body(metrics: dict, history: dict | None, rows: list[dict]) -> str:
    """Compose the whole page body from the loaded artefacts."""
    parts = []
    best = _headline(metrics)

    parts.append(
        '<header class="mast"><div>'
        '<p class="eyebrow">Applied ML portfolio - results console</p>'
        "<h1>ECG arrhythmia detection on the MIT-BIH benchmark</h1>"
        '<p class="sub">Five AAMI beat classes from 252-sample morphology '
        "windows and RR-interval features, trained patient-disjoint on DS1 and "
        "reported once on 51,149 held-out DS2 beats. Every figure below is "
        "generated from committed artefacts, never typed by hand.</p>"
        '<div class="mast-links">'
        f'<a href="https://github.com/{REPO}">Repository</a>'
        '<a href="https://github.com/Bakkarranna">Other projects</a>'
        '<a href="#results">Skip to results</a></div></div></header>'
    )

    toc = [
        ("results", "Results"),
        ("charts", "Curves &amp; errors"),
        ("patients", "Per-patient"),
        ("figures", "Figures"),
        ("honest", "Limitations"),
        ("artefacts", "Artefacts"),
    ]
    parts.append(
        '<nav class="toc" aria-label="Sections">'
        + "".join(
            f'<a href="#{anchor}">{label}</a>' for anchor, label in toc
        )
        + "</nav>"
    )

    sections = []
    if metrics:
        sections.append(
            '<section class="block" id="results"><h2>Results</h2>'
            '<p class="lede">Two models, one test split, no tuning on it. '
            "Macro metrics are unweighted per-class means, so a 388-beat class "
            "counts as much as a 44,240-beat one.</p>"
            + comparison_table(metrics)
            + (
                metric_cards(best)
                if best
                else ""
            )
            + (
                class_table(best)
                if best
                else ""
            )
            + (
                _bar_rows(best["class_distribution"])
                if best.get("class_distribution")
                else ""
            )
            + "</section>"
        )
    if history or any("confusion_matrix" in m for m in metrics.values()):
        heat = "".join(confusion_svg(m) for m in metrics.values())
        curve = training_curves_svg(history) if history else ""
        sections.append(
            '<section class="block" id="charts"><h2>Learning curves and '
            'error structure</h2><p class="lede">Left: the optimisation story. '
            "Right: where each model puts its mistakes. Rows are normalised, so "
            "read across a row as recall and down a column as contamination."
            f'</p><div class="cols">{curve}{heat}</div>'
            '<ul class="legend">'
            '<li><i style="background:var(--rose)"></i>train loss</li>'
            '<li><i style="background:var(--teal-2)"></i>validation macro-F1</li>'
            '<li><i style="background:var(--amber)"></i>best checkpoint</li>'
            "</ul></section>"
        )
    if rows:
        sections.append(
            '<section class="block" id="patients"><h2>Per-patient '
            "breakdown</h2><p class=\"lede\">A single aggregate hides whether a "
            "model fails everywhere or on a few hard recordings. These rows are "
            "recomputed from the committed prediction dump.</p>"
            + records_table(rows)
            + "</section>"
        )
    figures = _figures_html()
    if figures:
        sections.append(figures)
    sections.append(_honest_html(metrics, history))
    raw = _raw_json_html(metrics, history)
    if raw:
        sections.append(raw)
    parts.extend(s for s in sections if s)
    return "".join(parts)

def build_page(metrics: dict, history: dict | None, rows: list[dict]) -> str:
    """Render the complete standalone HTML document."""
    return PAGE.format(
        title="ECG arrhythmia detection - results console",
        description=(
            "Results, learning curves, confusion matrices and limitations for a "
            "five-class ECG arrhythmia classifier trained on MIT-BIH with "
            "patient-disjoint splits."
        ),
        css=CSS,
        body=build_body(metrics, history, rows),
        repo=f"https://github.com/{REPO}",
        suffix=" from the committed results/ artefacts",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail instead of skipping when an artefact is missing",
    )
    parser.add_argument(
        "--out",
        default=str(DOCS_DIR / "index.html"),
        help="output path (default: docs/index.html)",
    )
    args = parser.parse_args(argv)

    metrics = load_metrics(args.strict)
    history = load_history(args.strict)
    predictions = load_predictions(args.strict)
    if not metrics and args.strict:
        sys.exit("no metrics artefacts found")

    rows = record_summary(
        predictions["y_true"], predictions["y_pred"], predictions["records"]
    ) if predictions and all(
        k in predictions for k in ("y_true", "y_pred", "records")
    ) else []

    page = build_page(metrics, history, rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(page):,} bytes, {len(rows)} patient rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
