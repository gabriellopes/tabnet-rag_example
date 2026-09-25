# tests/src/kg/test_builder.py
import os
import pandas as pd
import pytest
from app.src.kg.builder import KnowledgeGraphBuilder


def test_knowledge_graph_builder_instantiation():
    # Mock do DataFrame resultante do cruzamento das 4 bases
    mock_df = pd.DataFrame(
        [
            {
                "cod_mun_6": "354890",
                "ano": 2025,
                "pop": 250000,
                "total_casos": 45,
                "taxa_incidencia": 18.0,
                "cnes_count": 12,
                "perc_acomp_saude": 85.5,
                "perc_gestantes_prenatal_em_dia": 91.2,
            }
        ]
    )

    builder = KnowledgeGraphBuilder()
    graph = builder.build_from_dataframe(mock_df)

    # Garante que gerou triplas
    assert len(graph) > 0

    # Testa exportação temporária
    test_output = "data/processed/test_graph.ttl"
    builder.export(test_output)

    assert os.path.exists(test_output)

    # Limpeza
    if os.path.exists(test_output):
         os.remove(test_output)