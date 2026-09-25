# app/src/readers/dbc.py
import pandas as pd
from pyreaddbc import read_dbc
from app.src.readers.base import BaseReader

class DBCReader(BaseReader):
    def __init__(self, file_path: str, encoding: str = "iso-8859-1"):
        self.file_path = file_path
        self.encoding = encoding

    def load_schema(self) -> list[str]:
        """Reads file and returns list of column headers."""
        df = read_dbc(self.file_path, encoding=self.encoding)
        return list(df.columns)

    def load_data(self) -> pd.DataFrame:
        """Loads full DBC dataset into a Pandas DataFrame."""
        return read_dbc(self.file_path, encoding=self.encoding)