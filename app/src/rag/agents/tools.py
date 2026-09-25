# app/src/rag/agents/tools.py
from typing import List, Dict, Any
from app.src.kg.sparql import SPARQLEngine

class GraphTools:
    def __init__(self, graph_path: str = "data/processed/graph.ttl"):
        self.engine = SPARQLEngine(graph_path=graph_path)

    def get_ontology_schema(self) -> str:
        return """
        Prefixes:
        PREFIX sus: <http://example.org/sus/ontology#>
        PREFIX ibge: <http://example.org/ibge/ontology#>
        PREFIX cnes: <http://example.org/cnes/ontology#>
        PREFIX pbf: <http://example.org/pbf/ontology#>
        PREFIX inst_ibge: <http://example.org/ibge/municipality/>

        Classes:
        - ibge:Municipality (ibge:hasCode6 xsd:string)
        - sus:HealthSnapshot (
            sus:forMunicipality -> ibge:Municipality,
            ibge:totalPopulation -> xsd:integer,
            sus:totalCases -> xsd:integer,
            sus:incidenceRatePer100k -> xsd:float,
            cnes:facilityCount -> xsd:integer,
            pbf:overallHealthComplianceRate -> xsd:float,
            pbf:prenatalComplianceRate -> xsd:float
          )
        """

    def execute_sparql(self, query_string: str) -> List[Dict[str, Any]]:
        return self.engine.query(query_string)