# tests/src/readers/test_dbc.py
import os
import pytest
from app.src.readers.dbc import DBCReader

# Path to your actual local .dbc file
SINAN_FILE_PATH = "data/raw/sinan/cancer_trab/CANCBR25.dbc" 

@pytest.mark.skipif(not os.path.exists(SINAN_FILE_PATH), reason="Raw SINAN file not found localy")
def test_load_schema_dbc():
    reader = DBCReader(SINAN_FILE_PATH)
    schema = reader.load_schema()
    
    assert isinstance(schema, list)
    assert len(schema) > 0
    print(f"\n[SINAN Schema ({len(schema)} columns)]:\n{schema}")