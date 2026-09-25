# app/src/readers/csv.py
import pandas as pd
from app.src.etl.readers.base import BaseReader

class CSVReader(BaseReader):
    def __init__(self, file_path: str, encoding: str = "utf-8", sep: str = ","):
        self.file_path = file_path
        self.encoding = encoding
        self.sep = sep

    def load_schema(self) -> list[str]:
        """Reads only the header line to return column names quickly."""
        df = pd.read_csv(self.file_path, encoding=self.encoding, sep=self.sep, nrows=0)
        return list(df.columns)

    def load_data(self) -> pd.DataFrame:
        """Loads full CSV dataset into a Pandas DataFrame."""
        return pd.read_csv(self.file_path, encoding=self.encoding, sep=self.sep)