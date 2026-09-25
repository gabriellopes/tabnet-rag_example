# app/src/readers/dbf.py
import pandas as pd
from dbfread import DBF
from app.src.etl.readers.base import BaseReader

class DBFReader(BaseReader):
    def __init__(self, file_path: str, encoding: str = "iso-8859-1"):
        self.file_path = file_path
        self.encoding = encoding

    def load_schema(self) -> list[str]:
        """Reads field names from header without loading records into memory."""
        table = DBF(self.file_path, load=False, encoding=self.encoding)
        return table.field_names

    def load_data(self) -> pd.DataFrame:
        """Loads records into a DataFrame."""
        table = DBF(self.file_path, encoding=self.encoding)
        return pd.DataFrame(iter(table))