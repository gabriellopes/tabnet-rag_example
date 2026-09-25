# app/src/plugins/finetuner/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseFineTuner(ABC):
    """Abstract Base Class for fine-tuning strategies in the AI-SUS pipeline."""

    def __init__(self, model_id: str, output_dir: str = "models/finetuned"):
        self.model_id = model_id
        self.output_dir = output_dir

    @abstractmethod
    def prepare_dataset(self, dataset_path: str) -> Any:
        """Loads and formats input dataset for model consumption."""
        pass

    @abstractmethod
    def train(
        self,
        dataset_path: str,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes the fine-tuning procedure and returns metrics."""
        pass

    @abstractmethod
    def save_adapter(self, path: Optional[str] = None) -> None:
        """Saves trained adapter/weights to disk."""
        pass