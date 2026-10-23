# Indoor Temperature Analysis

An MMA3001 Numerical Methods and Machine Learning project investigating **how well hour of day alone can predict indoor temperature**. The analysis compares four simple models using environmental sensor data and explores numerical integration, data quality, and loading performance.

## Project files

| File | Purpose |
| --- | --- |
| [`temperature_analysis.ipynb`](temperature_analysis.ipynb) | Main analysis notebook, including helper code, tests, benchmarking, and documentation generation |
| `5EnvSensor_MayToDec2024_180kRows.zip` | Compressed sensor dataset used by the notebook |
| `Sensor ID and Locations.xlsx` | Sensor location reference; not required by the notebook at runtime |
| `Indoor_Temperature_Analysis_7_Sections.ipynb` | Additional notebook; the instructions below target `temperature_analysis.ipynb` |
| [`LICENSE`](LICENSE) | MIT license |

## Requirements

Use Python 3.12 (the version recorded in the notebook metadata) with:

- NumPy **2.0 or newer**, because the code uses `np.trapezoid`
- pandas, Matplotlib, and requests
- JupyterLab to run the notebook locally
- pytest and pytest-html for testing and HTML reports
- pdoc for generated documentation

There is currently no `requirements.txt` or dependency lockfile in the repository. The commands below install the required packages directly.

## Getting started

```bash
git clone https://github.com/Nervermind13/MMA3001-Indoor-Temperature.git
cd MMA3001-Indoor-Temperature
python -m venv .venv
source .venv/bin/activate
python -m pip install "numpy>=2.0" pandas matplotlib requests jupyterlab pytest pytest-html pdoc
jupyter lab temperature_analysis.ipynb
```

On Windows, activate the virtual environment with `.venv\Scripts\activate` instead.

Select the virtual environment's Python kernel and run the notebook **from top to bottom**, with the repository root as its working directory. The first code cells create or overwrite `temperature_tools.py` and `test_temperature_tools.py` in that directory.

### Data settings

In the settings cell, replace the placeholder `ZIP_URL` with:

```python
ZIP_URL = "https://raw.githubusercontent.com/Nervermind13/MMA3001-Indoor-Temperature/main/5EnvSensor_MayToDec2024_180kRows.zip"
LOCAL_ZIP = "5EnvSensor_MayToDec2024_180kRows.zip"

SENSOR_ID = 6012002000869
START, END = "2024-05-01", "2025-01-01"
```

The loader tries the URL first and falls back to `LOCAL_ZIP` if the request fails or the response is not a ZIP archive. Keep the local ZIP in the working directory: the benchmark and profiler use it directly, even when the main download succeeds. pandas reads the compressed CSV without manual extraction.

The date filter includes `START` and excludes `END`. Timestamps are interpreted as Unix seconds in UTC and converted to `Australia/Melbourne` before dates and hours are calculated. The configured date range does not guarantee that data exists throughout that period.

### Google Colab

Clone the repository in a Colab session and change into it before running the analysis cells:

```python
!git clone https://github.com/Nervermind13/MMA3001-Indoor-Temperature.git
%cd /content/MMA3001-Indoor-Temperature
%pip install "numpy>=2.0" pandas matplotlib requests pytest pytest-html pdoc
```

Open or upload `temperature_analysis.ipynb`, then run its cells in order. If package installation updates libraries already loaded by Colab, restart the runtime and repeat the working-directory step.

## Analysis workflow

1. **Load data:** read only the `sensorid` and `jsondata` columns from the ZIP's CSV.
2. **Extract temperature:** parse the selected sensor's JSON readings and retain the `Temperature` variable.
3. **Clean readings:** remove missing values, non-numeric temperatures, duplicate timestamps, and temperatures outside −10 to 50 °C; apply the date filter.
4. **Select usable days:** retain days with at least 120 readings and coverage of all 24 hours.
5. **Split chronologically:** assign approximately 70% of usable days to training, 15% to validation, and the remainder to testing, without shuffling.
6. **Fit models:** calculate training-set hourly averages and compare the models below.
7. **Evaluate:** report validation RMSE, test RMSE, test MAE, parameter counts, and fitting time. Errors are evaluated on individual readings, in °C.
8. **Investigate:** compare piecewise boundaries, plot test-day hourly averages, calculate daily trapezoidal averages, and benchmark/profile data loading.

| Model | Description | Parameters |
| --- | --- | ---: |
| Hourly average | Lookup of the training mean for each hour | 24 |
| Linear baseline | Straight line fitted to the 24 training hourly means | 2 |
| Cubic | Third-degree polynomial fitted to the hourly means | 4 |
| Piecewise linear | Three lines with default boundaries at 06:00 and 08:00 | 6 |

The piecewise fit includes boundary hours in both adjoining segments. The later segment overwrites the earlier prediction at each shared boundary; continuity is not enforced.

## Generated outputs

Running the notebook produces:

- `temperature_tools.py`: reusable loading, cleaning, modelling, and metric functions.
- `test_temperature_tools.py`: 17 unit tests.
- `model_comparison.png`: model curves compared with hourly averages from test days.
