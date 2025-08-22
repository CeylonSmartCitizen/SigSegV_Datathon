# Ceylon Smart Citizen Datathon Project
## Team SigSegV

## Project Overview

This project was developed for the Rootcode Datathon 2025 under the theme "Ceylon Smart Citizen." Our solution leverages AI and data-driven decision-making to optimize government services in Sri Lanka by predicting service completion times and staffing requirements to improve efficiency and citizen satisfaction.

## Directory Structure

- `data/`
	- `raw/`: Contains raw training data (`bookings_train.csv`, `staffing_train.csv`, `tasks.csv`)
	- `evaluation/`: Contains test input files for both tasks
	- `processed/`: (Reserved for processed data)
- `models/`: Trained model files for both tasks (`task1_completion_time_model.pkl`, `task2_staffing_model.pkl`)
- `notebooks/`: Jupyter notebooks for model development (`Task1Model.ipynb`, `Task2Model.ipynb`)
- `src/`: Source code for model inference and prediction (`model_inference.py`)
- `submissions/`: Output prediction CSVs for submission
- `docs/`: Documentation and supporting files

## Tasks

### Task 1: Predict Service Completion Time

- **Data**: Uses `bookings_train.csv` for training and `task1_test_inputs.csv` for evaluation.
- **Model**: Trained using features such as appointment time, number of documents, queue number, and task type (one-hot encoded).
- **Notebook**: See `notebooks/Task1Model.ipynb` for data exploration, feature engineering, and model training.
- **Output**: Predictions saved as `submissions/task1_predictions.csv`.

### Task 2: Predict Staffing Requirements

- **Data**: Uses `staffing_train.csv` for training and `task2_test_inputs.csv` for evaluation.
- **Model**: Trained using date-based features (day of week, month, year, etc.), section encoding, and lag features.
- **Notebook**: See `notebooks/Task2Model.ipynb` for data exploration, feature engineering, and model training.
- **Output**: Predictions saved as `submissions/task2_predictions.csv`.

## How to Run Inference

1. Ensure all dependencies are installed (see below).
2. Place the trained model files in the `models/` directory.
3. Place the test input files in `data/evaluation/`.
4. Run the inference script:
	 ```bash
	 python src/model_inference.py
	 ```
	 This will generate prediction CSVs in the `submissions/` directory.

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib (for notebooks)

Install dependencies with:
```bash
pip install -r requirements.txt
```
*(Create a `requirements.txt` if not present.)*

## Team Members

- [Add your team members here]

## Documentation

- See `docs/DatathonDocumentation-SigSegV.pdf` for detailed methodology and results.

