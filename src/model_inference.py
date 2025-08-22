# Add src directory to sys.path for local imports
import os
import sys
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)
# Add src directory to sys.path for local imports
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
def save_predictions(predictions, output_path, id_data=None):
    """Save predictions to a CSV file, optionally including an ID column."""
    import numpy as np
    import pandas as pd
    if predictions is None:
        print(f"No predictions to save for {output_path}.")
        return
    
    # Create output dataframe
    if id_data is not None:
        output_df = pd.DataFrame({'row_id': id_data, 'prediction': predictions})
    else:
        output_df = pd.DataFrame({'prediction': predictions})
    
    output_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path} ({output_df.shape[0]} rows).")
def generate_predictions(model, data, task_name):
    """Generate predictions using the given model and data."""
    if model is None or data is None:
        print(f"Cannot generate predictions for {task_name}: model or data missing.")
        return None
    try:
        predictions = model.predict(data)
        print(f"Generated predictions for {task_name}: {len(predictions)} rows.")
        return predictions
    except Exception as e:
        print(f"Error generating predictions for {task_name}: {e}")
        return None
# Paths to model files (absolute paths)
task1_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'task1_completion_time_model.pkl')
task2_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'task2_staffing_model.pkl')

# Paths to test input files
task1_test_input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'evaluation', 'task1_test_inputs.csv')
task2_test_input_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'evaluation', 'task2_test_inputs.csv')

# Note: preprocessing and feature_engineering modules don't exist yet
# from preprocessing import preprocess_data
# from feature_engineering import feature_engineer

def preprocess_data(df):
    """Preprocessing function to transform test data to match training data format."""
    print(f"Preprocessing data with shape: {df.shape}")
    
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # For Task 1: Convert basic test format to training format
    if 'task_id' in processed_df.columns:
        # Parse time components
        processed_df['appointment_time'] = processed_df['time']
        
        # Add missing columns with default values (you may need to adjust these)
        processed_df['num_documents'] = 1  # Default value
        processed_df['queue_number'] = 1   # Default value
        
        # Remove columns that aren't needed for prediction
        if 'row_id' in processed_df.columns:
            processed_df = processed_df.drop('row_id', axis=1)
        if 'date' in processed_df.columns:
            processed_df = processed_df.drop('date', axis=1)
        if 'time' in processed_df.columns:
            processed_df = processed_df.drop('time', axis=1)
    
    # For Task 2: Basic preprocessing
    elif 'section_id' in processed_df.columns:
        # Remove row_id and date for now
        if 'row_id' in processed_df.columns:
            processed_df = processed_df.drop('row_id', axis=1)
        if 'date' in processed_df.columns:
            processed_df = processed_df.drop('date', axis=1)
    
    print(f"After basic preprocessing: {processed_df.shape}")
    return processed_df

def feature_engineer(df):
    """Feature engineering function to create features expected by the model."""
    print(f"Feature engineering data with shape: {df.shape}")
    
    processed_df = df.copy()
    
    # For Task 1: Create task indicator columns (one-hot encoding)
    if 'task_id' in processed_df.columns:
        # Get unique task IDs and create dummy variables
        task_dummies = pd.get_dummies(processed_df['task_id'], prefix='task')
        processed_df = pd.concat([processed_df, task_dummies], axis=1)
        processed_df = processed_df.drop('task_id', axis=1)
        
        # Process appointment_time if it exists
        if 'appointment_time' in processed_df.columns:
            # Convert time to numerical format (minutes since midnight)
            try:
                time_parts = processed_df['appointment_time'].str.split(':')
                hours = time_parts.str[0].astype(int)
                minutes = time_parts.str[1].astype(int)
                processed_df['appointment_time'] = hours * 60 + minutes
            except:
                # If conversion fails, use a default value
                processed_df['appointment_time'] = 540  # 9:00 AM in minutes
    
    # For Task 2: Create section indicator columns
    elif 'section_id' in processed_df.columns:
        section_dummies = pd.get_dummies(processed_df['section_id'], prefix='section')
        processed_df = pd.concat([processed_df, section_dummies], axis=1)
        processed_df = processed_df.drop('section_id', axis=1)
    
    print(f"After feature engineering: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns)}")
    return processed_df

# Move all imports to the top
import os
import pickle
import pandas as pd
import sys

def load_model(model_path):
    """Load a pickle model from the given path."""
    try:
        import joblib
        # Try loading with joblib first (more robust for sklearn models)
        model = joblib.load(model_path)
        return model
    except:
        try:
            # Fallback to pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return None

def get_task1_model():
    return load_model(task1_model_path)

def get_task2_model():
    return load_model(task2_model_path)

# Example usage (to be replaced with actual inference logic)

# Step 1: Load test input CSVs
import pandas as pd

def load_test_inputs():
    """Load test input CSVs for both tasks."""
    try:
        task1_df = pd.read_csv(task1_test_input_path)
        print(f"Loaded Task 1 test inputs: {task1_df.shape}")
    except Exception as e:
        print(f"Error loading Task 1 test inputs: {e}")
        task1_df = None
    try:
        task2_df = pd.read_csv(task2_test_input_path)
        print(f"Loaded Task 2 test inputs: {task2_df.shape}")
    except Exception as e:
        print(f"Error loading Task 2 test inputs: {e}")
        task2_df = None
    return task1_df, task2_df

def preprocess_test_inputs(task1_df, task2_df):
    """Preprocess test inputs using training pipeline."""
    if task1_df is not None:
        task1_df = preprocess_data(task1_df)
        task1_df = feature_engineer(task1_df)
        print(f"Preprocessed Task 1 test inputs: {task1_df.shape}")
    if task2_df is not None:
        task2_df = preprocess_data(task2_df)
        task2_df = feature_engineer(task2_df)
        print(f"Preprocessed Task 2 test inputs: {task2_df.shape}")
    return task1_df, task2_df

if __name__ == "__main__":
    # Load models
    task1_model = get_task1_model()
    task2_model = get_task2_model()
    print("Task 1 Model Loaded:", type(task1_model))
    print("Task 2 Model Loaded:", type(task2_model))

    # Step 1: Load test inputs
    task1_df, task2_df = load_test_inputs()
    
    # Store original dataframes to preserve row_id
    task1_df_original = task1_df.copy() if task1_df is not None else None
    task2_df_original = task2_df.copy() if task2_df is not None else None

    # Step 2: Preprocess test inputs
    task1_df, task2_df = preprocess_test_inputs(task1_df, task2_df)

    # Step 3: Generate predictions
    task1_preds = generate_predictions(task1_model, task1_df, "Task 1")
    task2_preds = generate_predictions(task2_model, task2_df, "Task 2")

    # Step 4: Save predictions as CSV files
    submissions_dir = os.path.join(os.path.dirname(__file__), '..', 'submissions')
    task1_output_path = os.path.join(submissions_dir, 'task1_predictions.csv')
    task2_output_path = os.path.join(submissions_dir, 'task2_predictions.csv')

    # If test input has an ID column, include it; otherwise, just save predictions
    task1_id_data = task1_df_original['row_id'] if task1_df is not None and 'row_id' in task1_df_original.columns else None
    task2_id_data = task2_df_original['row_id'] if task2_df is not None and 'row_id' in task2_df_original.columns else None

    save_predictions(task1_preds, task1_output_path, id_data=task1_id_data)
    save_predictions(task2_preds, task2_output_path, id_data=task2_id_data)
