# tests/src/readers/test_csv.py
import os
import pytest
from app.src.etl.readers.csv import CSVReader

CSV_FILE_PATH = "data/raw/mds/pbf/2025.csv"

@pytest.mark.skipif(not os.path.exists(CSV_FILE_PATH), reason="MDS/PBF CSV file not found locally")
def test_load_schema_csv():
    reader = CSVReader(CSV_FILE_PATH)
    schema = reader.load_schema()
    
    assert isinstance(schema, list)
    assert len(schema) > 0
    print(f"\n[MDS/PBF Schema ({len(schema)} columns)]:\n{schema}")