# app/src/readers/base.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseReader(ABC):
    """Abstract Base Class for all data ingestion readers."""

    @abstractmethod
    def load_schema(self) -> list[str]:
        """Extract and return list of column headers without loading the full dataset."""
        pass

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load and return the complete dataset as a Pandas DataFrame."""
        pass