# Uber Availability Prediction (15-Minute Forecast)

Predicting short-term Uber ride availability in a given area using machine learning. The model forecasts whether rides will be available — and estimated wait times — within the next **15 minutes**, helping both riders plan trips and Uber optimize driver allocation.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Objectives](#objectives)
3. [Dataset](#dataset)
4. [Project Architecture](#project-architecture)
5. [Tech Stack](#tech-stack)
6. [Installation & Setup](#installation--setup)
7. [Usage](#usage)
8. [Feature Engineering](#feature-engineering)
9. [Modeling Approach](#modeling-approach)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Results](#results)
12. [Future Work](#future-work)
13. [Contributing](#contributing)
14. [License](#license)

---

## Problem Statement

In urban environments, Uber ride availability fluctuates rapidly due to demand surges, weather, events, and time of day. Riders frequently face long wait times or unavailability with no advance warning. This project builds a predictive system that forecasts Uber availability for the **next 15 minutes** at a given location, enabling proactive trip planning and smarter fleet management.

---

## Objectives

- Predict whether Uber rides will be available in a specific zone within the next 15 minutes.
- Estimate the expected wait time (in minutes) for the next available ride.
- Identify the key factors that drive ride availability fluctuations.
- Provide an actionable, deployable prediction pipeline.

---

## Dataset

### Primary Data Sources

| Source | Description |
|---|---|
| **Uber Movement / Ride Data** | Historical trip records including pickup/dropoff coordinates, timestamps, and trip duration. |
| **Weather API** | Temperature, precipitation, wind speed, and visibility at the time of each record. |
| **Event Data** | Local events (concerts, sports, conferences) sourced from public event APIs. |
| **Calendar Features** | Holidays, weekends, day-of-week, hour-of-day derived from timestamps. |

### Data Fields (Example)

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Pickup request time |
| `latitude` | float | Pickup latitude |
| `longitude` | float | Pickup longitude |
| `zone_id` | string | Geohash or hex grid zone identifier |
| `temperature` | float | Temperature in °C at request time |
| `precipitation` | float | Rainfall in mm |
| `is_holiday` | bool | Whether the day is a public holiday |
| `hour` | int | Hour of the day (0–23) |
| `day_of_week` | int | Day of week (0=Mon, 6=Sun) |
| `active_drivers` | int | Number of active drivers in the zone |
| `pending_requests` | int | Unfulfilled ride requests in the zone |
| `availability` | bool | **Target** — ride available within 15 min |
| `wait_time_min` | float | **Target** — estimated wait in minutes |

> **Note:** If using a public dataset (e.g., Uber Pickups from Kaggle or NYC TLC data), document the source and any preprocessing steps applied.

---

## Project Architecture

```
uber-availability-prediction/
│
├── data/
│   ├── raw/                  # Original unprocessed datasets
│   ├── processed/            # Cleaned and feature-engineered data
│   └── external/             # Weather, event, and holiday data
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_feature_eng.ipynb  # Feature engineering pipeline
│   ├── 03_modeling.ipynb     # Model training and comparison
│   └── 04_evaluation.ipynb   # Evaluation and result analysis
│
├── src/
│   ├── data_loader.py        # Data ingestion and cleaning
│   ├── feature_engine.py     # Feature engineering functions
│   ├── model.py              # Model training and inference
│   ├── predict.py            # 15-minute prediction pipeline
│   └── utils.py              # Helper functions
│
├── models/                   # Saved model artifacts (.pkl, .joblib)
├── reports/
│   └── figures/              # Charts, plots, confusion matrices
│
├── app/
│   ├── app.py                # Flask/Streamlit demo application
│   └── templates/            # HTML templates (if Flask)
│
├── requirements.txt
├── config.yaml               # Hyperparameters and path configs
├── README.md
└── LICENSE
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy, GeoPandas |
| Visualization | Matplotlib, Seaborn, Folium |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Deep Learning (optional) | TensorFlow / PyTorch (LSTM for time-series) |
| Deployment | Flask / Streamlit / FastAPI |
| Scheduling | APScheduler / Cron (for live retraining) |
| Version Control | Git, GitHub |

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/uber-availability-prediction.git
cd uber-availability-prediction

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place raw data files in data/raw/

# 5. Run the preprocessing pipeline
python src/data_loader.py

# 6. Train the model
python src/model.py

# 7. Launch the demo app
streamlit run app/app.py
```

---

## Usage

### Quick Prediction (CLI)

```bash
python src/predict.py --lat 40.7484 --lon -73.9857 --time "2026-05-12T14:30:00"
```

**Output:**

```
Zone: dr5ru7 | Availability: YES | Est. Wait: 3.2 min | Confidence: 91.4%
```

### Python API

```python
from src.predict import predict_availability

result = predict_availability(
    latitude=40.7484,
    longitude=-73.9857,
    timestamp="2026-05-12T14:30:00"
)

print(result)
# {'available': True, 'wait_time_min': 3.2, 'confidence': 0.914}
```

---

## Feature Engineering

Key features constructed for the 15-minute prediction window:

| Feature | Description |
|---|---|
| `hour_sin`, `hour_cos` | Cyclical encoding of hour |
| `day_of_week_sin`, `day_of_week_cos` | Cyclical encoding of weekday |
| `rolling_demand_15m` | Rolling count of requests in the past 15 min |
| `rolling_supply_15m` | Rolling count of completed trips in the past 15 min |
| `supply_demand_ratio` | `active_drivers / (pending_requests + 1)` |
| `surge_indicator` | Binary flag when demand exceeds supply by 2x |
| `weather_severity` | Composite score from rain, wind, and visibility |
| `is_rush_hour` | Flag for 7–9 AM and 5–8 PM weekdays |
| `event_proximity` | Distance (km) to nearest active event venue |
| `geohash_encoded` | Target-encoded zone identifier |
| `lag_availability_15m` | Availability label from 15 minutes ago |

---

## Modeling Approach

### Classification Task — Will a ride be available?

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient-boosted trees (primary model) |
| LightGBM | Fast gradient boosting for large datasets |
| LSTM (optional) | Sequence model over time-series features |

### Regression Task — How long is the wait?

A secondary regression head predicts `wait_time_min` using the same feature set, trained with MAE loss.

### Training Strategy

- **Train/Validation/Test Split:** 70/15/15 chronological split (no data leakage).
- **Cross-Validation:** TimeSeriesSplit with 5 folds.
- **Hyperparameter Tuning:** Optuna or GridSearchCV.
- **Class Imbalance Handling:** SMOTE / class weights for the classification task.

---

## Evaluation Metrics

### Classification (Availability)

| Metric | Purpose |
|---|---|
| **Accuracy** | Overall correctness |
| **Precision** | How often "available" predictions are correct |
| **Recall** | How many actual available rides are captured |
| **F1-Score** | Harmonic mean of precision and recall |
| **AUC-ROC** | Discrimination ability across thresholds |

### Regression (Wait Time)

| Metric | Purpose |
|---|---|
| **MAE** | Average prediction error in minutes |
| **RMSE** | Penalizes large errors |
| **R² Score** | Proportion of variance explained |

---

## Results

> *Update this section after training.*

| Model | Accuracy | F1-Score | AUC-ROC | MAE (min) |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| **XGBoost** | — | — | — | — |
| LightGBM | — | — | — | — |

### Feature Importance (Top 5)

1. `supply_demand_ratio`
2. `rolling_demand_15m`
3. `hour_sin`
4. `weather_severity`
5. `surge_indicator`

---

## Future Work

- **Real-time streaming** — Integrate Apache Kafka for live prediction updates every minute.
- **Multi-city generalization** — Train on data from multiple cities and add city-level embeddings.
- **Competitor comparison** — Extend to predict Lyft/Bolt availability alongside Uber.
- **Graph Neural Networks** — Model spatial dependencies between adjacent zones.
- **Mobile app integration** — Push notifications when predicted availability drops below a threshold.
- **Explainability dashboard** — SHAP-based explanations for each prediction.

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a Pull Request.

Please ensure code follows PEP 8 and includes docstrings.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Author:** *Your Name*
**Contact:** *your.email@example.com*
**Last Updated:** May 2026
