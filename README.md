# Predicting indoor temperature from the hour of day

MMA3001 Numerical Methods and Machine Learning, individual project.

## 1. Overview
This project builds and compares simple numerical models that predict the temperature of one room from the hour of day,
using building sensor data. Everything runs from one notebook, `temperature_analysis.ipynb`.

## 2. Engineering problem
Heating and cooling systems waste energy when they are scheduled without knowing how a room actually behaves over a day.
A model of the normal daily temperature pattern can be used as a baseline for HVAC scheduling, and for spotting days when a room
behaves unusually (a large gap between predicted and measured temperature).

**Intended use:** a quick, explainable baseline for one room, not a replacement for a full building thermal model.

**Chosen scope:** one sensor, hour of day as the only input.
The sensor is `6012002000869`, listed in `Sensor_ID_and_Locations.xlsx` as building 60, room G.38 ("Compactus_upper", label EMS107).
Between 1 May and 14 June 2024 it has 5,927 readings over 45 days, of which 44 days are complete enough to use.

## 3. Data
| Item | Detail |
|---|---|
| `5EnvSensor_MayToDec2024_180kRows.zip` | Zip holding one CSV with 179,447 rows from 5 environmental sensors (350 MB unzipped) |
| `Sensor_ID_and_Locations.xlsx` | Maps sensor IDs to buildings and rooms. Used only to identify the room |
| Source | Provided by the MMA3001 teaching team for this project. Not redistributed beyond this assignment |

Despite the file name, the chosen sensor only has data from 29 February to 14 June 2024, so no summer or spring conditions are covered.

## 4. Inputs
| Input | Format | Units and domain |
|---|---|---|
| `sensorid` column | Integer sensor ID | Must exist in the data |
| `jsondata` column | JSON text: a list of variables, each with `name`, `unit` and `values` (`time`, `value`) | `time` is Unix seconds in UTC |
| Temperature `value` | Number | °C. Values outside -10 to 50 are treated as sensor faults |
| `SENSOR_ID`, `START`, `END` | Notebook settings | Sensor ID; dates as `YYYY-MM-DD` |
| `ZIP_URL` | Raw GitHub link to the zip | Must be a raw link, not the normal page link |

## 5. Outputs
| Output | Meaning |
|---|---|
| Model comparison table | Accuracy (RMSE and MAE in °C), number of parameters, fit time and interpretability for four models |
| Sensitivity table | How much the piecewise boundary choice changes validation error |
| `model_comparison.png` | Unseen test days plotted against each model |
| Daily trapezoid averages | Time-weighted average temperature per day, checked against the simple mean |
| Loader benchmark and profiler | Time and memory of the original and improved data loader |
| `test_report.html` | pytest results |
| `docs/` | HTML documentation generated from the docstrings (pdoc) |

## 6. How invalid or missing inputs are handled
| Problem | What happens |
|---|---|
| Sensor ID not in the data | `ValueError` listing the sensors that exist |
| `sensorid` or `jsondata` column missing | `KeyError` naming the column |
| A row with unreadable JSON | Row skipped |
| Missing, non-numeric or duplicate readings | Dropped |
| Temperature outside -10 to 50 °C | Dropped as a sensor fault |
| No readings left after cleaning | `ValueError` |
| Days with fewer than 120 readings or missing hours | Excluded from modelling |
| No complete days, or fewer than 3 days | `ValueError` |
| GitHub download fails or is not a zip (wrong link, Git LFS pointer) | Message printed, local zip used instead |

## 7. Quick start
1. Put the zip, `Sensor_ID_and_Locations.xlsx` and `temperature_analysis.ipynb` in one folder.
   In Colab: `!git clone https://github.com/USER/REPO` then `%cd REPO`.
2. In the settings cell, replace `USER` and `REPO` in `ZIP_URL` and check the branch name (`main` or `master`).
3. Run the notebook from top to bottom. It writes `temperature_tools.py` and `test_temperature_tools.py`, runs the tests and builds the docs.
4. From a terminal instead: `pip install -r requirements.txt`, then `python -m pytest -v` and `python -m pdoc temperature_tools -o docs`.

Tested on Python 3.12. Needs NumPy 2.0 or newer (the code uses `np.trapezoid`).

## 8. Repository contents
| File | Purpose |
|---|---|
| `temperature_analysis.ipynb` | The analysis. Run this |
| `temperature_tools.py` | Helper functions with docstrings (written by the notebook) |
| `test_temperature_tools.py` | pytest tests (written by the notebook) |
| `test_report.html` | Test report |
| `docs/` | HTML documentation |
| `5EnvSensor_MayToDec2024_180kRows.zip` | Data (the 350 MB csv inside is not committed) |
| `Sensor_ID_and_Locations.xlsx` | Sensor locations |
| `requirements.txt`, `LICENSE`, `.gitignore` | Setup files |

## 9. Method
1. **Load** the zip directly with pandas (two columns only).
2. **Extract** the temperature readings for one sensor from the JSON, convert Unix time to Melbourne time, and clean.
3. **Select good days:** at least 120 readings (about 144 are expected at one reading every 10 minutes) and all 24 hours present.
4. **Split by date, in time order:** 70% train, 15% validation, 15% test (30, 7 and 7 days). No shuffling, so the test days come after the training days.
5. **Fit models on training days only.** Each model is built from the average temperature at each hour of the day:
   - *Hourly average (lookup):* the 24 averages themselves, no curve.
   - *Linear (baseline):* one straight line through the 24 averages.
   - *Cubic:* a third-degree polynomial.
   - *Piecewise linear:* a separate straight line for hours 0 to 6, 6 to 8 and 8 to 23, to capture the night, the morning rise and the day.
6. **Evaluate** on individual readings from validation and test days, using RMSE and MAE.
7. **Numerical integration:** the daily average temperature by the trapezoidal rule.

## 10. Results
Errors are in °C on individual readings from days the models never saw.

| Model | Parameters | Validation RMSE | Test RMSE | Test MAE | Interpretability |
|---|---|---|---|---|---|
| Hourly average (lookup) | 24 | 0.688 | 0.543 | 0.393 | Very easy: it is the data |
| Linear (baseline) | 2 | 0.904 | 0.863 | 0.651 | Very easy: one slope |
| Cubic | 4 | 0.848 | 0.838 | 0.628 | Hard: coefficients have no physical meaning |
| Piecewise linear | 6 | 0.685 | 0.512 | 0.361 | Easy: night, morning and day slopes |

- The piecewise model has the lowest error. The daily pattern probably has a sharp morning rise rather than a smooth curve, which would explain why the cubic barely improves on a straight line.
- The hourly lookup is almost as accurate (validation RMSE 0.688 against 0.685). The case for the piecewise model is similar accuracy with 6 parameters instead of 24, and a shape that can be explained physically. It is not clearly the most accurate.
- Fitting takes well under a millisecond for every model, so runtime does not separate them.
- Moving the piecewise boundaries changes validation RMSE only between 0.685 and 0.722 over five tested settings, with (6, 8) best.
- The trapezoid daily average differs from the simple mean by 0.0045 °C on average (0.0121 °C at most), as expected when readings are close to evenly spaced.

## 11. Validation
- **What was validated:** the ability of each model to predict temperatures on days it was not fitted on.
- **Why this method:** the split is by date and in time order, so the test is like predicting the future from the past. A random split would leak information between neighbouring readings.
- **Metrics:** RMSE (penalises large misses) and MAE (typical miss), both in °C.
- **Did validation influence model selection?** Yes. The piecewise boundaries were first chosen by looking at the data, then compared on validation days (sensitivity table). The test days were only used for the final comparison.
- **Software checks:** 17 pytest tests, including an analytical check (the exact average of a straight line), a perfect-fit check for the piecewise model, and tests for each invalid input in section 6.
- **What it cannot establish:** whether the models work for other rooms, other seasons or years, because only one sensor and 7 test days from late autumn and early winter were available.

## 12. Performance
The original loader read every column, looped over every row with `iterrows()` and parsed the JSON of all five sensors.
The improved loader reads two columns and parses only the chosen sensor. Notebook section 11 times both and section 12 profiles the improved one.
In a test run the improved loader was roughly 4 to 5 times faster but used only about 14% less memory, because the JSON column is most of the data.
The profiler showed most of the remaining time goes into reading and unzipping the CSV, with JSON parsing second. Timings depend on your machine, so re-run the notebook for your own numbers.
A FLOPs estimate is not meaningful here, because fitting 24 points is trivial and the run time is dominated by reading and parsing text.

## 13. Limitations and future work
- One sensor, one room, and about six weeks of data.
- Only 7 test days, so small differences between close models are within normal day-to-day variation.
- The piecewise boundaries were chosen by looking at the data, then checked on validation days.
- Hour of day is the only input. Weather, occupancy and HVAC state are not used, so unusual days are not explained.
- Future work: add outdoor temperature as an input, test on the other sensors and seasons, and compare against a small machine learning model.

## 14. AI use and reflection
*Edit every line marked [CONFIRM] so it is true for you, then delete the markers. Only claim checks you actually did.*

**Tools used.** Claude (Anthropic) and ChatGPT (OpenAI). [CONFIRM: add any other tool]

**What they were used for.**
- ChatGPT: I first tried it to load the dataset from a zip in a GitHub repository, and it did not produce working code. [CONFIRM: say which parts of the original notebook (the analysis steps up to the sampling-interval check) were written with which tool.]
- Claude: debugging the data loading, restructuring the code into helper functions, writing the unit tests, input validation, the benchmark and profiler cells, the documentation set-up, and drafts of this README and the repository files.

**Approximate level of AI contribution.** The large majority of the code was AI-generated [CONFIRM: about __%]. The engineering question, the choice of data, the sensor and the overall comparison were directed by me [CONFIRM]. The written report is my own work [CONFIRM].

**Why AI was used.** To get working code faster than I could write it alone, and to use good practice I had not used before: separate helper functions, unit tests, docstrings and generated documentation. [CONFIRM or edit]

**How AI output was checked.**
- I ran every cell and the test suite myself. [CONFIRM]
- After the code was restructured, I compared its results with the printed results of my original notebook: the same 5,927 readings, 44 good days, a 30/7/7 split, and the same trapezoid differences (0.0045 and 0.0121 °C). [CONFIRM you checked this]
- I timed the original and improved loaders instead of relying on the claim that the new one was faster, which showed the speed-up was real but the memory saving small.
- The tests were also written by AI, so passing tests only show the code does what the tests expect. I checked that the analytical test is correct by hand: the average of y = 2t + 1 on [0, 4] is 5. [CONFIRM]
- I can explain each function and each step in this notebook without AI. [CONFIRM: only keep this if true]

**Decisions that stayed my responsibility.** [CONFIRM and edit] Choosing the project and sensor; using hour of day as the only input; splitting by date in time order; the 120-readings rule for a usable day; which models to compare; accepting the piecewise model only as "similar accuracy with fewer parameters" rather than the clear winner; and what to claim in the report.

**Errors, limitations and unhelpful suggestions from AI.**
1. The original notebook never read from GitHub. It loaded a file from Colab's `/content` folder, which only exists after a manual upload, so the first generic suggestions for downloading from GitHub did not address the real cause.
2. A plot title in the original notebook named sensor 6012002000326 while the data plotted was sensor 6012002000869.
3. The piecewise model fitted the boundary hours (6 and 8) twice because its segments overlapped. I kept this behaviour so my results did not change, and I disclose it as a known quirk.
4. The piecewise boundaries were first chosen by looking at all of the data, which is a mild form of data leakage that the AI-generated workflow did not flag.
5. The first unit test for the piecewise model failed because the test data the AI wrote had a jump that no straight line could fit. The test was wrong, not the code.
6. The AI first described the faster loader as taking "a few seconds, not minutes" without timing the original. When I measured both, the improvement was about 4 to 5 times in speed and only about 14% in memory.

**How AI affected my understanding and workflow.** [CONFIRM: write 2 to 3 honest sentences in your own words, for example what you understood better after reading the tests, and anything you still find hard to explain.]

## 15. Licence
MIT. See `LICENSE`.

## 16. Acknowledgements
Data provided by the MMA3001 teaching team. Libraries: pandas, NumPy, Matplotlib, requests, pytest, pytest-html, pdoc.
