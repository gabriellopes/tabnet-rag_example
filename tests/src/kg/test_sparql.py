# tests/src/kg/test_sparql.py
import os
import pytest
from app.src.etl.aggregator import DatasetAggregator
from app.src.kg.builder import KnowledgeGraphBuilder
from app.src.kg.sparql import SPARQLEngine

SINAN_FILE = "data/raw/sus/sinan/cancer_trab/CANCBR25.dbc"
IBGE_FILE  = "data/raw/sus/ibge/POPSBR25/POP25.dbf"
CNES_FILE  = "data/raw/sus/cnes/STBA2512.dbc"
PBF_FILE   = "data/raw/mds/pbf/2025.csv"
GRAPH_TTL  = "data/processed/graph.ttl"

@pytest.mark.skipif(
    not (os.path.exists(SINAN_FILE) and os.path.exists(IBGE_FILE) and os.path.exists(CNES_FILE) and os.path.exists(PBF_FILE)),
    reason="One or more raw data files missing locally"
)
def test_full_kg_build_and_sparql_query():
    # 1. Aggregate real data sources
    aggregator = DatasetAggregator(SINAN_FILE, IBGE_FILE, CNES_FILE, PBF_FILE)
    df_unified = aggregator.load_and_aggregate()
    assert not df_unified.empty

    # 2. Build and export Knowledge Graph to data/processed/graph.ttl
    builder = KnowledgeGraphBuilder(ontology_path="app/src/kg/ontology.ttl")
    graph = builder.build_from_dataframe(df_unified)
    assert len(graph) > 0
    
    builder.export(GRAPH_TTL)
    assert os.path.exists(GRAPH_TTL)

    # 3. Instantiate SPARQLEngine on the persistent graph
    engine = SPARQLEngine(graph_path=GRAPH_TTL)

    # 4. Test custom SPARQL query (extract top 5 municipalities by disease incidence)
    custom_query = """
    PREFIX sus: <http://example.org/sus/ontology#>
    PREFIX ibge: <http://example.org/ibge/ontology#>
    PREFIX cnes: <http://example.org/cnes/ontology#>

    SELECT ?cod_mun ?pop ?cases ?incidence ?facilities
    WHERE {
        ?mun a ibge:Municipality ;
             ibge:hasCode6 ?cod_mun .
        
        ?snapshot sus:forMunicipality ?mun ;
                  ibge:totalPopulation ?pop ;
                  sus:totalCases ?cases ;
                  sus:incidenceRatePer100k ?incidence ;
                  cnes:facilityCount ?facilities .
    }
    ORDER BY DESC(?incidence)
    LIMIT 5
    """
    
    results = engine.query(custom_query)
    print(f"\n=== SPARQL QUERY RESULTS ({len(results)} municipalities) ===")
    for res in results:
        print(res)

    assert len(results) > 0

    # 5. Test pre-packaged helper method using a code present in the graph results
    sample_code = str(results[0]["cod_mun"])
    summary = engine.get_municipality_summary(sample_code)
    print(f"\n=== MUNICIPALITY SUMMARY FOR {sample_code} ===")
    print(summary)

    assert len(summary) >= 1