
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import mcp_server


@pytest.fixture
def sample_sales(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")

    data = pd.DataFrame({
        "date": dates,
        "vehicles_sold": [100000] * 12 + [110000] * 12,
    })

    monkeypatch.setattr(mcp_server, "load_sales", lambda: data)


def test_annual_sales(sample_sales):
    result = mcp_server.get_annual_sales(2024)

    assert result["vehicles_sold"] == 1200000
    assert result["months_available"] == 12


def test_compare_sales(sample_sales):
    result = mcp_server.compare_sales(2024, 2025)

    assert result["change_pct"] == 10.0


def test_latest_sales(sample_sales):
    result = mcp_server.get_latest_sales()

    assert result["month"] == "2025-12"
    assert result["vehicles_sold"] == 110000


def test_missing_year(sample_sales):
    result = mcp_server.get_annual_sales(2023)

    assert "error" in result
