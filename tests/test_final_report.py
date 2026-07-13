from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

FINAL_OUTPUT = Path("outputs/final_report")


def test_final_report_script_runs_and_generates_required_files() -> None:
    subprocess.run(
        [sys.executable, "scripts/generate_final_report_outputs.py"],
        check=True,
    )

    required = [
        "final_project_summary.csv",
        "final_stage_timeline.csv",
        "final_key_metrics.csv",
        "final_model_comparison.csv",
        "final_limitations.csv",
        "final_resume_metrics.csv",
        "plots/final_sabr_before_after_filtering.png",
        "plots/final_sabr_vs_heston_rmse.png",
        "plots/final_pipeline_timeline.png",
        "plots/final_warning_reduction.png",
    ]
    for relative_path in required:
        assert (FINAL_OUTPUT / relative_path).exists()


def test_required_final_docs_exist() -> None:
    required = [
        "docs/research_note.md",
        "docs/project_summary.md",
        "docs/website_case_study.md",
        "docs/technical_faq.md",
    ]

    for path in required:
        assert Path(path).exists()


def test_readme_contains_key_sections_without_duplicate_stage_3_5() -> None:
    readme = Path("README.md").read_text()

    for section in [
        "## Key Result",
        "## What This Project Proves",
        "## Headline Metrics",
        "## Reproduce",
        "## Limitations",
    ]:
        assert section in readme
    assert readme.count("Stage 3.5") == 0
    assert "Remaining optional work: historical validation" in readme


def test_public_docs_do_not_contain_career_packaging_terms() -> None:
    public_docs = [
        "README.md",
        "docs/research_note.md",
        "docs/project_summary.md",
        "docs/website_case_study.md",
        "docs/technical_faq.md",
    ]
    forbidden_terms = [
        "resume bullets",
        "interview defense",
        "recruiter",
        "hiring",
        "job application",
    ]

    for path in public_docs:
        text = Path(path).read_text().lower()
        for term in forbidden_terms:
            assert term not in text


def test_final_report_csvs_have_required_columns() -> None:
    expectations = {
        "final_project_summary.csv": [
            "metric",
            "value",
            "source_stage",
            "interpretation",
        ],
        "final_stage_timeline.csv": [
            "stage",
            "name",
            "implemented_scope",
            "key_output",
            "main_finding",
        ],
        "final_key_metrics.csv": [
            "category",
            "metric",
            "value",
            "interpretation",
        ],
        "final_model_comparison.csv": [
            "model_or_variant",
            "universe",
            "rmse_iv",
            "weighted_rmse_iv",
            "warning_summary",
            "interpretation",
        ],
        "final_limitations.csv": [
            "limitation",
            "impact",
            "mitigation_or_future_work",
        ],
        "final_resume_metrics.csv": [
            "metric",
            "value",
            "resume_safe",
            "caveat",
        ],
    }

    for filename, columns in expectations.items():
        assert list(pd.read_csv(FINAL_OUTPUT / filename).columns) == columns
