# test_configs.py
from app.src.utils.config_loader import ConfigLoader

rag_cfg = ConfigLoader.load_rag_config()
reranker_cfg = ConfigLoader.load_reranker_config()
finetuner_cfg = ConfigLoader.load_finetuner_config()

print("RAG Embedding Model:", rag_cfg["embedding"]["model_name"])
print("Active Reranker:", reranker_cfg["active_plugin"])
print("Active Fine-Tuner:", finetuner_cfg["active_plugin"])