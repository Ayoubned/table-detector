import pytest
from TableDetector import TableDetector

@pytest.fixture
def detector():
    return TableDetector()

def test_successful_table_detection(detector):
    results = detector.predict("testimages/Table.png")
    assert len(results["boxes"]) > 0, "No tables detected in the invoice image."
    
def test_successful_bank_document_detection(detector):
    results = detector.predict("testimages/BankDoc.png")
    assert len(results["boxes"]) > 0, "No tables detected in the bank document."

def test_no_table_in_image(detector):
    results = detector.predict("testimages/puppy.png")
    assert len(results["boxes"]) == 0, "Unexpected tables detected in the image without tables."

def test_invalid_image_path(detector):
    with pytest.raises(FileNotFoundError):
        detector.predict("testimages/non_existent.jpg")
