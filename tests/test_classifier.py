import pytest
from werkzeug.datastructures import FileStorage
from unittest.mock import MagicMock
import os
from unittest.mock import patch
import joblib
import src.classifier as classifier

SAMPLE_FILE_DIR = "tests/test_sample_files"

@pytest.mark.parametrize("filename,expected", [
    ("test_license.jpg", "license"),
    ("test_bank_statement.pdf", "bank statement"),
    ("test_invoice.pdf", "invoice")
])

def test_classify_file(filename, expected):
    """
    Loads a sample file and tests the classification
    """
    with open(f"{SAMPLE_FILE_DIR}/{filename}", 'rb') as fp:
        data = FileStorage(fp)
        result = classifier.classify_file(data)

    assert result == expected


def test_invalid_file_format():
    """
    Tests the classification of an invalid file format
    """
    invalid_file = MagicMock()
    invalid_file.filename = "test_invalid_file.xyz"
    invalid_file.read.return_value = b"Invalid content"

    result = classifier.classify_file(invalid_file)
    
    assert result == "Invalid file type or format"


def test_empty_file():
    """
    Tests the classification of an empty file
    """
    mock_file = MagicMock()
    mock_file.filename = "test_empty.pdf"
    mock_file.read.return_value = b""  

    result = classifier.classify_file(mock_file)

    assert result == "Invalid file type or format"


@patch.dict(os.environ, {"MODEL_FILE": "invalid_path"})
def test_model_load_failure():
    """
    Tests the failure of loading the model
    """
    with pytest.raises(Exception):
        classifier.model = joblib.load(os.getenv("MODEL_FILE"))