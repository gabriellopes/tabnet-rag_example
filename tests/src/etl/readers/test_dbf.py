# tests/src/readers/test_dbf.py
import os
import pytest
from app.src.etl.readers.dbf import DBFReader

# Path to your actual local .dbf file
IBGE_FILE_PATH = "data/raw/sus/ibge/POPSBR25/POP25.dbf" 

@pytest.mark.skipif(not os.path.exists(IBGE_FILE_PATH), reason="Raw IBGE file not found locally")
def test_load_schema_dbf():
    reader = DBFReader(IBGE_FILE_PATH)
    schema = reader.load_schema()
    
    assert isinstance(schema, list)
    assert len(schema) > 0
    print(f"\n[IBGE Schema ({len(schema)} columns)]:\n{schema}")