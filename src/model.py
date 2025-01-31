import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import logging
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Tuple


load_dotenv()

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Paths and configurations from environment variables
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = os.getenv("DATA_FILE", BASE_DIR / "files/training_data/classifier_training_set.csv") # Synthetic data to train classifier 
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = os.getenv("MODEL_FILE", BASE_DIR / "models/text_classification_model.joblib")
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2)) 
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 97)) 


def load_data(filepath: str) -> Tuple[List[str], List[str]]:
    """
    Loads dataset from a CSV file and validates it

    parameters: 
        filepath (str): Path to the CSV file

    returns:
        Tuple[List[str], List[str]]: A tuple containing the list of texts and labels
    """
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    
    if 'Text' not in df.columns or 'Label' not in df.columns:
        logging.error("CSV file must contain 'Text' and 'Label' columns.")
        raise ValueError("CSV file must contain 'Text' and 'Label' columns.")
    
    logging.info(f"Successfully loaded {len(df)} records from {filepath}")
    return df['Text'].tolist(), df['Label'].tolist()


def train_and_evaluate(texts: List[str], labels: List[str]) -> Pipeline:
    """
    Trains and evaluates the text classification model.

    parameters: 
        texts (List[str]): List of text samples
        labels (List[str]): List of corresponding labels

    returns:
        Pipeline: Trained text classification model
    """
    # Split dataset, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Define the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()), 
        ('classifier', LogisticRegression(max_iter=500))  # Increased iterations for convergence
    ])

    # Train model
    logging.info("Training the model...")
    pipeline.fit(X_train, y_train)

    # Evaluate model accuracy
    y_pred = pipeline.predict(X_test)
    accuracy_percentage = accuracy_score(y_test, y_pred) * 100
    logging.info(f"Model Accuracy: {accuracy_percentage:.2f}%")

    return pipeline


def save_model(model: Pipeline, filepath: str):
    """
    Saves the trained model to a file.
    
    parameters:
        model (Pipeline): Trained text classification model
        filepath (str): Path to save the model file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model_path = MODEL_DIR / "text_classification_model.joblib"

    joblib.dump(model, model_path)
    logging.info(f"Model saved at {model_path}")


if __name__ == "__main__":
    try:
        texts, labels = load_data(DATA_FILE)
        model = train_and_evaluate(texts, labels)
        save_model(model, MODEL_FILE)
    except Exception as e:
        logging.exception("An error occurred during execution.")
