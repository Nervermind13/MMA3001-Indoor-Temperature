"""Helper functions for the indoor temperature project.

Everything the notebook does that can be tested on its own lives here.
Inputs are pandas DataFrames / NumPy arrays; temperatures are in degrees C,
times are Unix seconds (UTC) unless stated otherwise.

Typical use::

    raw = pd.read_csv("data.zip", compression="zip", usecols=["sensorid", "jsondata"])
    data = extract_temperatures(raw, sensor_id=6012002000869)
    data = clean_readings(data, "2024-05-01", "2025-01-01")
    data = select_good_days(data)
"""
import json

import numpy as np
import pandas as pd

# Columns the raw data must contain
REQUIRED_COLUMNS = ("sensorid", "jsondata")


def extract_temperatures(raw, sensor_id):
    """Pull the temperature readings for one sensor out of the raw JSON rows.

    Args:
        raw: DataFrame with columns "sensorid" and "jsondata" (a JSON string
            holding a list of variables, each with a list of timestamped values).
        sensor_id: Sensor ID to keep (same type as raw["sensorid"], e.g. int).

    Returns:
        DataFrame with columns "unix_time" and "temperature".
        Rows with unreadable JSON are skipped, not raised as errors.

    Raises:
        KeyError: If a required column is missing from raw.
        ValueError: If the sensor is not in the data, or has no temperature readings.
    """
    # Check the input shape before doing any work, so errors are clear
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise KeyError(f"Input is missing column(s): {missing}")
    available = sorted(raw["sensorid"].unique())
    if sensor_id not in available:
        raise ValueError(f"Sensor {sensor_id} not found. Available sensors: {available}")

    rows = []
    # Only this sensor's rows are parsed: JSON parsing is the slow part
    for text in raw.loc[raw["sensorid"] == sensor_id, "jsondata"]:
        try:
            variables = json.loads(text)
            for item in variables:
                # Each row also holds battery, humidity, CO2 etc.; keep temperature only
                if item["variable"]["name"] == "Temperature":
                    for reading in item["values"]:
                        rows.append((reading["time"], reading["value"]))
        except (ValueError, TypeError, KeyError):
            continue  # bad or unexpected row: skip it
    if not rows:
        raise ValueError(f"Sensor {sensor_id} has no readable temperature readings")
    return pd.DataFrame(rows, columns=["unix_time", "temperature"])


def clean_readings(data, start, end, timezone="Australia/Melbourne", valid_range=(-10, 50)):
    """Convert times, drop bad/duplicate readings and keep one date range.

    Args:
        data: DataFrame from extract_temperatures().
        start, end: Date strings; keeps start <= time < end (local time).
        timezone: Local time zone used for the "hour" and "date" columns.
        valid_range: (low, high) in degrees C. Readings outside this range are
            treated as sensor faults and dropped.

    Returns:
        New DataFrame sorted by time with extra columns
        "datetime", "date" and "hour" (0-23).

    Raises:
        ValueError: If no readings are left after cleaning.
    """
    data = data.copy()  # never change the caller's DataFrame
    # errors="coerce" turns text like "bad" into NaN so dropna() can remove it
    data["temperature"] = pd.to_numeric(data["temperature"], errors="coerce")
    data = data.dropna().drop_duplicates("unix_time")
    # Physically impossible indoor temperatures are sensor faults
    low, high = valid_range
    data = data[(data["temperature"] >= low) & (data["temperature"] <= high)]
    # Unix seconds are UTC; convert to local time so "hour" matches the clock on the wall
    data["datetime"] = pd.to_datetime(data["unix_time"], unit="s", utc=True).dt.tz_convert(timezone)
    data = data[(data["datetime"] >= start) & (data["datetime"] < end)]
    data = data.sort_values("datetime")
    if data.empty:
        raise ValueError("No readings left after cleaning (check dates, units and valid_range)")
    data["date"] = data["datetime"].dt.date
    data["hour"] = data["datetime"].dt.hour
    return data


def select_good_days(data, min_readings=120):
    """Keep only days with enough readings that also cover all 24 hours.

    Args:
        data: DataFrame from clean_readings() (needs "date", "hour", "temperature").
        min_readings: Minimum number of readings a day must have. At one reading
            every 10 minutes a full day has about 144.

    Returns:
        DataFrame containing only the readings from days that pass both checks.

    Raises:
        ValueError: If no day passes both checks.
    """
    # One row per day: how many readings, and how many different hours they cover
    per_day = data.groupby("date").agg(readings=("temperature", "size"),
                                       hours=("hour", "nunique"))
    good = per_day[(per_day["readings"] >= min_readings) & (per_day["hours"] == 24)].index
    if len(good) == 0:
        raise ValueError("No days have enough readings covering all 24 hours")
    return data[data["date"].isin(good)]


def split_days(days, train_frac=0.70, val_frac=0.15):
    """Split dates into train / validation / test in time order (no shuffling).

    Keeping time order means the test days come after the training days, like
    predicting the future from the past.

    Args:
        days: Iterable of dates.
        train_frac, val_frac: Fractions for train and validation; the rest is test.

    Returns:
        (train_days, val_days, test_days) as lists.

    Raises:
        ValueError: If there are fewer than 3 days (each split needs at least one).
    """
    days = sorted(days)
    n = len(days)
    if n < 3:
        raise ValueError("Need at least 3 days to make train, validation and test sets")
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))  # cut positions
    return days[:a], days[a:b], days[b:]


def fit_piecewise(hours, hourly_avg, boundaries=(6, 8)):
    """Fit a straight line to each of three parts of the day.

    Parts are hours [0, b1], [b1, b2] and [b2, 23] (inclusive, so the boundary
    hours belong to two parts and the later part overwrites the earlier one).

    Args:
        hours: Array of hours 0..23.
        hourly_avg: Average temperature at each hour (same length as hours).
        boundaries: (b1, b2) hours where the lines change.

    Returns:
        Array of 24 predicted temperatures, one per hour.
    """
    hours = np.asarray(hours)
    hourly_avg = np.asarray(hourly_avg, dtype=float)
    b1, b2 = boundaries
    pred = np.zeros(len(hours))
    for start, end in [(0, b1), (b1, b2), (b2, 23)]:
        mask = (hours >= start) & (hours <= end)               # hours in this part
        coeffs = np.polyfit(hours[mask], hourly_avg[mask], 1)  # slope and intercept
        pred[mask] = np.polyval(coeffs, hours[mask])
    return pred


def rmse(actual, predicted):
    """Root mean square error: the typical size of a prediction error.

    Squaring means big misses count more than small ones.

    Args:
        actual: Measured values (array-like).
        predicted: Model predictions, same length as actual.

    Returns:
        RMSE as a float, in the same units as the inputs (degrees C here).
    """
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual, predicted):
    """Mean absolute error: the average size of a prediction error.

    Args:
        actual: Measured values (array-like).
        predicted: Model predictions, same length as actual.

    Returns:
        MAE as a float, in the same units as the inputs (degrees C here).
    """
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.mean(np.abs(actual - predicted)))


def trapezoid_average(t_hours, values):
    """Time-weighted average of a signal using the trapezoidal rule.

    Unlike a simple mean, this accounts for uneven gaps between readings.

    Args:
        t_hours: Increasing times in hours (at least two points).
        values: Signal value at each time.

    Returns:
        Area under the curve divided by the total duration.

    Raises:
        ValueError: If there are fewer than two different times.
    """
    t_hours, values = np.asarray(t_hours, float), np.asarray(values, float)
    if len(t_hours) < 2 or t_hours[-1] == t_hours[0]:
        raise ValueError("Need at least two different times")
    area = np.trapz(values, t_hours)  # area under the curve (value x hours)
    return float(area / (t_hours[-1] - t_hours[0]))