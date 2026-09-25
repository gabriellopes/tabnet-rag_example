# main.py
import os
from app.src.etl.aggregator import DatasetAggregator
from app.src.kg.builder import KnowledgeGraphBuilder

SINAN_FILE = "data/raw/sinan/cancer_trab/CANCBR25.dbc"
IBGE_FILE  = "data/raw/ibge/POPSBR25/POP25.dbf"
CNES_FILE  = "data/raw/sus/cnes/STBA2512.dbc"
PBF_FILE   = "data/raw/mds/pbf/2025.csv"
OUTPUT_TTL = "data/processed/graph.ttl"

def main():
    print("=== STARTING AI-SUS KNOWLEDGE GRAPH PIPELINE ===")
    
    print("\n1. Aggregating raw datasets into unified schema...")
    aggregator = DatasetAggregator(SINAN_FILE, IBGE_FILE, CNES_FILE, PBF_FILE)
    df_unified = aggregator.load_and_aggregate()
    print(f"   ✓ Consolidated {len(df_unified)} intersecting municipalities.")

    print("\n2. Transforming DataFrame records into RDF triples...")
    builder = KnowledgeGraphBuilder(ontology_path="app/src/kg/ontology.ttl")
    graph = builder.build_from_dataframe(df_unified)

    print("\n3. Exporting Knowledge Graph...")
    builder.export(OUTPUT_TTL)
    print(f"   ✓ Successfully generated '{OUTPUT_TTL}' containing {len(graph)} triples.")

if __name__ == "__main__":
    main()