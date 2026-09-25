# app/src/plugins/finetuner/sft.py
import json

class SFTDatasetGenerator:
    """Generates SPARQL-to-Answer fine-tuning pairs for training/fine-tuning models."""

    @staticmethod
    def create_pair(user_prompt: str, sparql_query: str, ground_truth: str) -> dict:
        return {
            "messages": [
                {"role": "system", "content": "You are a SPARQL-fluent AI public health analyst for Brazil's SUS and MDS data."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": f"```sparql\n{sparql_query}\n```\n\n{ground_truth}"}
            ]
        }

    def export_dataset(self, pairs: list, output_path: str = "data/processed/sft_data.jsonl"):
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")