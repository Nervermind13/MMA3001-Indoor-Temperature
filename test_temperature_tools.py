"""Tests for temperature_tools. Run with:  pytest -v"""
import json

import numpy as np
import pandas as pd
import pytest

import temperature_tools as tt


def make_raw():
    """Tiny fake dataset: two sensors, one broken row."""
    good = json.dumps([{"variable": {"name": "Temperature"},
                        "values": [{"time": 1714521600, "value": 21.5}]},
                       {"variable": {"name": "Humidity"},
                        "values": [{"time": 1714521600, "value": 50}]}])
    other = json.dumps([{"variable": {"name": "Temperature"},
                         "values": [{"time": 1714521600, "value": 30.0}]}])
    return pd.DataFrame({"sensorid": [1, 2, 1],
                         "jsondata": [good, other, "not json at all"]})


def test_extract_keeps_only_requested_sensor_and_variable():
    """Only the chosen sensor's Temperature values come out (not humidity, not other sensors)."""
    out = tt.extract_temperatures(make_raw(), 1)
    assert len(out) == 1
    assert out["temperature"].iloc[0] == 21.5


def test_extract_skips_broken_json_without_crashing():
    """A row with unreadable JSON is skipped instead of crashing."""
    out = tt.extract_temperatures(make_raw(), 1)  # sensor 1 has a broken row
    assert list(out.columns) == ["unix_time", "temperature"]


def test_clean_removes_duplicates_and_missing():
    """Repeated timestamps and non-numeric values are removed."""
    raw = pd.DataFrame({"unix_time": [1714521600, 1714521600, 1714522200],
                        "temperature": [20.0, 20.0, "bad"]})
    out = tt.clean_readings(raw, "2024-05-01", "2024-06-01")
    assert len(out) == 1  # duplicate removed, "bad" -> NaN -> dropped


def test_clean_hour_uses_melbourne_time():
    """Hours are in local time: 00:00 UTC is 10:00 in Melbourne in May."""
    # 2024-05-01 00:00 UTC is 10:00 in Melbourne (UTC+10 in May)
    raw = pd.DataFrame({"unix_time": [1714521600], "temperature": [20.0]})
    out = tt.clean_readings(raw, "2024-05-01", "2024-06-01")
    assert out["hour"].iloc[0] == 10


def test_split_is_in_time_order_with_no_overlap():
    """Train, validation and test days are in time order and never overlap."""
    days = list(range(20))[::-1]  # deliberately unsorted
    train, val, test = tt.split_days(days)
    assert len(train) + len(val) + len(test) == 20
    assert max(train) < min(val) and max(val) < min(test)
    assert set(train).isdisjoint(val) and set(val).isdisjoint(test)


def test_rmse_and_mae_known_values():
    """RMSE and MAE match values worked out by hand."""
    actual = [1, 2, 3]
    predicted = [2, 2, 5]           # errors: 1, 0, 2
    assert tt.mae(actual, predicted) == pytest.approx(1.0)
    assert tt.rmse(actual, predicted) == pytest.approx(np.sqrt(5 / 3))


def test_trapezoid_of_constant_is_that_constant():
    """A constant signal must average to itself, even with uneven spacing."""
    t = np.array([0, 0.3, 1.0, 2.5])        # uneven spacing on purpose
    assert tt.trapezoid_average(t, np.full(4, 22.0)) == pytest.approx(22.0)


def test_trapezoid_of_straight_line_matches_analytical_answer():
    """Analytical check: the exact average of y = 2t + 1 on [0, 4] is 5."""
    # y = 2t + 1 on [0, 4]: exact average is the midpoint value 5
    t = np.array([0, 1.0, 1.5, 4.0])
    assert tt.trapezoid_average(t, 2 * t + 1) == pytest.approx(5.0)


def test_trapezoid_rejects_single_point():
    """One reading is not enough to integrate, so a ValueError is raised."""
    with pytest.raises(ValueError):
        tt.trapezoid_average([1.0], [20.0])


def test_piecewise_recovers_data_that_is_already_piecewise_linear():
    """If the data really is three straight lines, the fit should be (almost) exact."""
    hours = np.arange(24)
    # continuous curve with kinks exactly at hours 6 and 8 (slopes 0.1, 1.0, 0.5)
    truth = np.piecewise(
        hours.astype(float),
        [hours <= 6, (hours > 6) & (hours <= 8), hours > 8],
        [lambda h: 20.0 + 0.1 * h,
         lambda h: 20.6 + 1.0 * (h - 6),
         lambda h: 22.6 + 0.5 * (h - 8)],
    )
    pred = tt.fit_piecewise(hours, truth, boundaries=(6, 8))
    assert tt.rmse(truth, pred) < 1e-9


def test_piecewise_returns_one_value_per_hour():
    """The piecewise model gives exactly 24 predictions."""
    hours = np.arange(24)
    pred = tt.fit_piecewise(hours, np.linspace(20, 25, 24))
    assert pred.shape == (24,)


# ---- invalid / missing / unsupported inputs ----
def test_unknown_sensor_gives_clear_error():
    """Asking for a sensor that is not in the data raises a clear ValueError."""
    with pytest.raises(ValueError, match="not found"):
        tt.extract_temperatures(make_raw(), 999)


def test_missing_column_gives_clear_error():
    """Input without the required columns raises a KeyError."""
    with pytest.raises(KeyError):
        tt.extract_temperatures(pd.DataFrame({"sensorid": [1]}), 1)


def test_clean_drops_physically_impossible_temperatures():
    """A 99 C indoor reading is treated as a sensor fault and removed."""
    raw = pd.DataFrame({"unix_time": [1714521600, 1714522200],
                        "temperature": [21.0, 99.0]})   # 99 C indoors = sensor fault
    out = tt.clean_readings(raw, "2024-05-01", "2024-06-01")
    assert list(out["temperature"]) == [21.0]


def test_clean_raises_if_nothing_left():
    """A date range with no data raises a ValueError instead of returning nothing."""
    raw = pd.DataFrame({"unix_time": [1714521600], "temperature": [21.0]})
    with pytest.raises(ValueError):
        tt.clean_readings(raw, "2030-01-01", "2030-02-01")   # wrong date range


def test_select_good_days_raises_if_no_day_is_complete():
    """If no day has enough readings, a ValueError is raised."""
    raw = pd.DataFrame({"unix_time": [1714521600, 1714522200], "temperature": [21.0, 21.5]})
    data = tt.clean_readings(raw, "2024-05-01", "2024-06-01")
    with pytest.raises(ValueError):
        tt.select_good_days(data)


def test_split_needs_at_least_three_days():
    """Fewer than 3 days cannot make three splits, so a ValueError is raised."""
    with pytest.raises(ValueError):
        tt.split_days([1, 2])