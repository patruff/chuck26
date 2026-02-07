"""Tests for training/training_report.py: report generation and cost estimation."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pytest

# Add training directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from training_report import (
    GPU_COSTS,
    ComparisonExample,
    TrainingReport,
)


# ---------------------------------------------------------------------------
# TrainingReport tests
# ---------------------------------------------------------------------------


def test_training_report_basic():
    """TrainingReport initializes with defaults."""
    report = TrainingReport()
    assert report.base_model == ""
    assert report.epochs == 0
    assert report.comparison_examples == []
    assert report.debug_log == []


def test_training_report_with_values():
    """TrainingReport stores provided values."""
    report = TrainingReport(
        base_model="test/model",
        output_model="output/model",
        dataset="test/dataset",
        epochs=5,
        batch_size=4,
        gpu_type="rtx3090",
    )
    assert report.base_model == "test/model"
    assert report.output_model == "output/model"
    assert report.dataset == "test/dataset"
    assert report.epochs == 5
    assert report.batch_size == 4
    assert report.gpu_type == "rtx3090"


def test_add_log():
    """add_log adds timestamped entries."""
    report = TrainingReport()
    report.add_log("Test message")

    assert len(report.debug_log) == 1
    assert "Test message" in report.debug_log[0]
    # Should have timestamp format
    assert "[" in report.debug_log[0] and "]" in report.debug_log[0]


def test_set_timing():
    """set_timing calculates duration correctly."""
    report = TrainingReport()
    start = time.time() - 3661  # 1h 1m 1s ago
    end = time.time()

    report.set_timing(start, end)

    assert report.training_duration_seconds > 3660
    assert report.training_duration_seconds < 3662
    assert "1h" in report.training_duration_human
    assert "1m" in report.training_duration_human


def test_set_timing_short():
    """set_timing formats short durations correctly."""
    report = TrainingReport()
    start = time.time() - 125  # 2m 5s ago
    end = time.time()

    report.set_timing(start, end)

    assert "2m" in report.training_duration_human
    assert "h" not in report.training_duration_human


def test_calculate_cost_rtx3090():
    """calculate_cost uses correct GPU pricing."""
    report = TrainingReport(gpu_type="rtx3090")
    report.training_duration_seconds = 3600  # 1 hour

    report.calculate_cost()

    assert report.gpu_cost_per_hour == 0.22
    assert report.estimated_cost == 0.22


def test_calculate_cost_a100():
    """calculate_cost works for A100."""
    report = TrainingReport(gpu_type="a100-80")
    report.training_duration_seconds = 7200  # 2 hours

    report.calculate_cost()

    assert report.gpu_cost_per_hour == 1.69
    assert report.estimated_cost == pytest.approx(3.38, rel=0.01)


def test_calculate_cost_full_gpu_name():
    """calculate_cost matches partial GPU names."""
    report = TrainingReport(gpu_type="NVIDIA GeForce RTX 4090")
    report.training_duration_seconds = 1800  # 30 min

    report.calculate_cost()

    assert report.gpu_cost_per_hour == 0.44
    assert report.estimated_cost == pytest.approx(0.22, rel=0.01)


def test_to_dict():
    """to_dict converts report to dictionary."""
    report = TrainingReport(
        base_model="test/model",
        epochs=3,
    )
    report.comparison_examples.append(
        ComparisonExample(
            input_title="Test",
            base_output="Base",
            finetuned_output="Fine",
            base_score=0.5,
            finetuned_score=0.8,
            improvement=0.3,
        )
    )

    data = report.to_dict()

    assert data["base_model"] == "test/model"
    assert data["epochs"] == 3
    assert len(data["comparison_examples"]) == 1
    assert data["comparison_examples"][0]["input_title"] == "Test"


def test_to_markdown():
    """to_markdown generates markdown report."""
    report = TrainingReport(
        base_model="test/model",
        output_model="test/output",
        epochs=3,
        gpu_type="rtx3090",
    )
    report.training_duration_human = "1h 30m"
    report.estimated_cost = 0.33

    md = report.to_markdown()

    assert "# Training Report" in md
    assert "test/model" in md
    assert "test/output" in md
    assert "1h 30m" in md
    assert "$0.33" in md


def test_to_markdown_with_comparison():
    """to_markdown includes comparison table."""
    report = TrainingReport(base_model="test")
    report.comparison_examples = [
        ComparisonExample(
            input_title="The Matrix",
            base_output="The Mattress",
            finetuned_output="The Mattress",
            base_score=0.5,
            finetuned_score=0.8,
            improvement=0.3,
        )
    ]

    md = report.to_markdown()

    assert "Model Comparison" in md
    assert "The Matrix" in md
    assert "The Mattress" in md
    assert "+0.30" in md or "0.30" in md


def test_save_json(tmp_path):
    """save writes JSON file."""
    report = TrainingReport(base_model="test", epochs=3)
    path = tmp_path / "report.json"

    report.save(path)

    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert data["base_model"] == "test"
    assert data["epochs"] == 3


def test_save_markdown(tmp_path):
    """save_markdown writes markdown file."""
    report = TrainingReport(base_model="test")
    path = tmp_path / "report.md"

    report.save_markdown(path)

    assert path.exists()
    content = path.read_text()
    assert "# Training Report" in content


# ---------------------------------------------------------------------------
# ComparisonExample tests
# ---------------------------------------------------------------------------


def test_comparison_example_basic():
    """ComparisonExample holds data correctly."""
    ex = ComparisonExample(
        input_title="Test Title",
        base_output="Base Out",
        finetuned_output="Fine Out",
        base_score=0.4,
        finetuned_score=0.7,
        improvement=0.3,
    )

    assert ex.input_title == "Test Title"
    assert ex.base_output == "Base Out"
    assert ex.finetuned_output == "Fine Out"
    assert ex.base_score == 0.4
    assert ex.finetuned_score == 0.7
    assert ex.improvement == 0.3


def test_comparison_example_defaults():
    """ComparisonExample has sensible defaults."""
    ex = ComparisonExample(
        input_title="Test",
        base_output="Base",
        finetuned_output="Fine",
    )

    assert ex.base_score == 0.0
    assert ex.finetuned_score == 0.0
    assert ex.improvement == 0.0


# ---------------------------------------------------------------------------
# GPU_COSTS tests
# ---------------------------------------------------------------------------


def test_gpu_costs_has_expected_gpus():
    """GPU_COSTS contains expected GPU options."""
    assert "rtx3090" in GPU_COSTS
    assert "rtx4090" in GPU_COSTS
    assert "a100-40" in GPU_COSTS
    assert "a100-80" in GPU_COSTS


def test_gpu_costs_reasonable_values():
    """GPU costs are in reasonable range."""
    for gpu, cost in GPU_COSTS.items():
        assert 0.1 < cost < 5.0, f"{gpu} cost {cost} seems unreasonable"


def test_rtx3090_cheapest():
    """RTX 3090 is among the cheapest options."""
    rtx3090_cost = GPU_COSTS["rtx3090"]
    assert rtx3090_cost <= 0.25
    # Should be cheaper than most other GPUs
    cheaper_count = sum(1 for cost in GPU_COSTS.values() if cost < rtx3090_cost)
    assert cheaper_count <= 2  # At most 2 GPUs cheaper
