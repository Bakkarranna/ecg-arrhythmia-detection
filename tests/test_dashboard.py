"""Tests for the static results dashboard (pure functions, no artefacts)."""

from __future__ import annotations

import re

import pytest

from ecg_arrhythmia import dashboard


@pytest.fixture
def metrics():
    """A minimal but structurally complete metrics dict for one model."""
    return {
        "macro_recall": 0.5101,
        "macro_precision": 0.4502,
        "macro_f1": 0.4588,
        "accuracy": 0.9688,
        "n_samples": 51149,
        "model": "BeatCNN(1D CNN + RR features)",
        "class_distribution": {"N": 44240, "S": 1837, "V": 3220, "F": 388, "Q": 1464},
        "confusion_matrix": {
            "labels": ["N", "S", "V", "F", "Q"],
            "matrix": [
                [35000, 7500, 400, 100, 1240],
                [900, 800, 40, 10, 87],
                [481, 60, 2890, 20, 60],
                [120, 40, 30, 150, 48],
                [240, 190, 60, 40, 934],
            ],
        },
        "per_class": {
            "N": {"precision": 0.95, "recall": 0.79, "f1-score": 0.86, "support": 44240},
            "S": {"precision": 0.47, "recall": 0.44, "f1-score": 0.45, "support": 1837},
            "V": {"precision": 0.89, "recall": 0.85, "f1-score": 0.87, "support": 3220},
            "F": {"precision": 0.31, "recall": 0.39, "f1-score": 0.34, "support": 388},
            "Q": {"precision": 0.61, "recall": 0.64, "f1-score": 0.62, "support": 1464},
        },
    }


@pytest.fixture
def history():
    """A short but faithful training history."""
    return {
        "epochs": [1, 2, 3, 4, 5],
        "train_loss": [0.88, 0.47, 0.39, 0.29, 0.28],
        "val_macro_f1": [0.26, 0.31, 0.35, 0.34, 0.33],
        "best_val_macro_f1": 0.35,
    }


@pytest.fixture
def rows():
    """Two per-record rollups, one clean and one with a dominant mistake."""
    return [
        {
            "record": "114",
            "n_beats": 1500,
            "accuracy": 1.0,
            "recall": {"N": 1.0, "V": 1.0},
            "errors": 0,
            "worst_error": "-",
        },
        {
            "record": "207",
            "n_beats": 800,
            "accuracy": 0.85,
            "recall": {"N": 0.9, "V": 0.6},
            "errors": 120,
            "worst_error": "V->N",
        },
    ]


def test_esc_neutralises_markup():
    assert dashboard.esc('<img src="x">&') == "&lt;img src=&quot;x&quot;&gt;&amp;"
    assert dashboard.esc(None) == ""
    assert dashboard.esc(51149) == "51149"


def test_fmt_renders_percentages_counts_and_plain_numbers():
    assert dashboard.fmt(0.4588) == "45.88%"
    assert dashboard.fmt(0.0) == "0.00%"
    assert dashboard.fmt(1.0) == "100.00%"
    assert dashboard.fmt(51149) == "51,149"
    assert dashboard.fmt(1234.5) == "1,234.5"
    assert dashboard.fmt(True) == "yes"


def test_heat_ramp_is_ordered_and_clamped():
    assert dashboard._heat(0.0) == "#f8fafc"
    assert dashboard._heat(1.0) == "#0f4c5c"
    assert dashboard._heat(-3.0) == "#f8fafc"
    assert dashboard._heat(7.0) == "#0f4c5c"
    # Brighter heat (a larger value) must move toward the dark end of the ramp.
    assert int(dashboard._heat(0.9)[1:3], 16) < int(dashboard._heat(0.1)[1:3], 16)


def test_bar_rows_order_by_count_and_sum_to_hundred(metrics):
    html = dashboard._bar_rows(metrics["class_distribution"])
    # Only the share column (the bar width is a duplicate of the same number).
    shares = [float(v) for v in re.findall(r'class="num">([0-9]+\.[0-9]{2})%</td>', html)]
    assert len(shares) == 5
    assert pytest.approx(sum(shares), abs=0.05) == 100.0
    # Biggest class first.
    assert html.index(">N<") < html.index(">Q<")


def test_record_summary_reports_accuracy_errors_and_worst_mistake():
    y_true = ["N", "N", "V", "V", "F"]
    y_pred = ["N", "S", "V", "N", "F"]
    records = ["100", "100", "100", "101", "101"]
    rows = dashboard.record_summary(y_true, y_pred, records)
    assert [r["record"] for r in rows] == ["100", "101"]
    hundred, oh_one = rows
    assert hundred["n_beats"] == 3
    assert hundred["errors"] == 1
    assert hundred["worst_error"] == "N->S"
    assert hundred["accuracy"] == pytest.approx(2 / 3)
    # Record 100 has two N beats (one right) and one V beat (right).
    assert hundred["recall"] == {"N": pytest.approx(0.5), "V": pytest.approx(1.0)}
    assert oh_one["errors"] == 1
    assert oh_one["worst_error"] == "V->N"
    assert oh_one["accuracy"] == pytest.approx(0.5)
    assert oh_one["recall"] == {"F": pytest.approx(1.0), "V": pytest.approx(0.0)}


def test_record_summary_marks_clean_records_with_a_dash():
    rows = dashboard.record_summary(["N", "N"], ["N", "N"], ["114", "114"])
    assert rows[0]["errors"] == 0
    assert rows[0]["worst_error"] == "-"
    assert rows[0]["accuracy"] == 1.0


def test_training_curves_svg_plots_every_epoch_and_marks_the_best(history):
    svg = dashboard.training_curves_svg(history)
    assert 'viewBox="0 0 660 260"' in svg
    # One epoch tick label per epoch when the run is short.
    for epoch in history["epochs"]:
        assert f'class="tick">{epoch}<' in svg
    assert svg.count("<polyline") == 2
    assert svg.count("<circle") == 2 * len(history["epochs"])
    # The best checkpoint line sits at the epoch with the best validation score.
    best = max(history["val_macro_f1"])
    assert f'class="bestlabel">best {best * 100:.1f}%<' in svg


def test_training_curves_svg_is_empty_without_a_history():
    assert dashboard.training_curves_svg({}) == ""
    assert dashboard.training_curves_svg({"epochs": [1]}) == ""


def test_confusion_svg_normalises_each_row(metrics):
    svg = dashboard.confusion_svg(metrics)
    for label in metrics["confusion_matrix"]["labels"]:
        assert f'class="lab">{label}<' in svg
    assert "predicted class" in svg
    assert "true class" in svg
    # Every row of the heatmap is normalised, so one row's cells sum to 100%
    # (to rounding), and a diagonal cell carries the row's recall.
    cells = re.findall(r'>([0-9]+\.[0-9])%<', svg)
    assert len(cells) == 25
    for start in range(0, 25, 5):
        assert pytest.approx(sum(float(c) for c in cells[start : start + 5]), abs=0.3) == 100.0
    assert "79.1" in cells[:5]  # 35000 / 44240 is the N recall


def test_metric_cards_show_four_headline_numbers(metrics):
    html = dashboard.metric_cards(metrics)
    assert html.count('class="kpis"') == 1
    # three headline metrics in accent cards, plus the plain sample-size card
    assert html.count('class="kpi"') == 3
    assert 'class="kpi kpi-plain"' in html
    assert "Macro recall" in html
    assert "51.01%" in html
    assert "45.88%" in html
    assert "51,149" in html
    assert "22 held-out patients" in html


def test_class_table_has_one_row_per_class(metrics):
    html = dashboard.class_table(metrics)
    assert html.count("<tr><th scope=\"row\">") == 5
    for code in ("N", "S", "V", "F", "Q"):
        assert f'c-{code}' in html
    assert "Precision" in html and "Support" in html


def test_comparison_table_sorts_by_macro_f1_and_marks_the_best(metrics):
    weak = dict(metrics, macro_recall=0.449, macro_precision=0.4186, macro_f1=0.4269,
                model="RandomForest(rr_intervals)")
    html = dashboard.comparison_table({"cnn": metrics, "baseline": weak})
    # Weakest first, so the strongest row is marked "best".
    assert html.index("RandomForest") < html.index("BeatCNN")
    assert 'class="best"' in html
    # The weakest model has a zero delta against itself; the best one gains.
    assert ">+0.0 pts<" in html
    assert "+3.2 pts" in html


def test_comparison_table_notes_training_cost(metrics):
    data = dict(metrics, epochs_ran=12, training_seconds=412.7)
    html = dashboard.comparison_table({"cnn": data})
    assert "12 epochs, checkpoint at best validation F1, 413s to train" in html


def test_records_table_lists_every_record(rows):
    html = dashboard.records_table(rows)
    assert html.count("<tr><th scope=\"row\">") == 2
    assert "record 114" in html
    assert "record 207" in html
    assert "1,500" in html
    assert ">V-&gt;N<" in html
    assert 'class="tag ok"' in html


def test_records_table_is_empty_without_rows():
    assert dashboard.records_table([]) == ""


def test_build_page_is_self_contained_and_reads_the_artefacts(
    metrics, history, rows, monkeypatch, tmp_path
):
    monkeypatch.setattr(dashboard, "DOCS_DIR", tmp_path)
    baseline = dict(metrics, macro_recall=0.449, macro_precision=0.4186,
                    macro_f1=0.4269, model="RandomForest(rr_intervals)")
    page = dashboard.build_page({"baseline": baseline, "cnn": metrics}, history, rows)
    assert page.startswith("<!doctype html>")
    assert "MIT-BIH" in page
    assert "51.01%" in page  # from the committed-style metric fixture
    assert "RandomForest" in page  # both model names surface, hand-written or not
    assert 'id="honest"' in page
    assert 'id="patients"' in page
    assert "<svg" in page
    # No CDN or build step: no external scripts, no web fonts, no network CSS.
    assert "<script" not in page
    assert "@import" not in page
    assert "cdn" not in page.lower()
    assert "googleapis" not in page.lower()


def test_main_writes_the_page_from_the_committed_artefacts(monkeypatch, tmp_path):
    monkeypatch.setattr(dashboard, "DOCS_DIR", tmp_path)
    out = tmp_path / "index.html"
    assert dashboard.main(["--out", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    assert "MIT-BIH" in page
    assert "51,149" in page
    assert (tmp_path / "figures").is_dir()
    assert out.stat().st_size > 20_000
