from werkzeug.datastructures import FileStorage
import joblib
import os
from pathlib import Path
from dotenv import load_dotenv
import sys
import logging 
from src.utils import text_extractor 

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = os.getenv("MODEL_FILE", BASE_DIR / "models/text_classification_model.joblib")


try:
    model = joblib.load(MODEL_FILE)
    logging.info(f"Model loaded from path: {MODEL_FILE}")
except Exception as e:
    logging.error(f"Failed to load model from path: {MODEL_FILE}. Error: {e}")
    raise


def extract_text_from_file(file: FileStorage) -> str:
    """
    Extract text from a file

    parameters:
        file: The file to extract text from

    returns:
        extracted_text: The text extracted from the file
    """
    try:
        file_bytes = file.read()
        file_extension = file.filename.split('.')[-1].lower()
        extracted_text = text_extractor(file_bytes, file_extension) 
    except Exception as e:
        logging.error(f"Error extracting text: {e}")
        return "No text extracted"

    return extracted_text


def classify_file(file: FileStorage) -> str:
    """
    Classify the file using the trained model

    parameters:
        file: The file to classify

    returns:   
        prediction: The predicted class of the file, either license, bank statement, or invoice
    """
    try:
        extracted_text = extract_text_from_file(file)
    except Exception as e:
        logging.error(f"Error processing: {e}")
        return "Invalid file type or format"

    try:
        prediction = model.predict([extracted_text]) 
    except Exception as e:
        logging.error(f"Error classifying: {e}")
        return "Failed to classify"
    
    return prediction[0] 
