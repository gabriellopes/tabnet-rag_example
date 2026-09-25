# app/src/kg/sparql.py
import os
from typing import Any, Dict, List
from rdflib import Graph


class SPARQLEngine:

    def __init__(self, graph_path: str = "data/processed/graph.ttl"):
        self.graph_path = graph_path
        self.graph = Graph()
        if os.path.exists(self.graph_path):
            self.graph.parse(self.graph_path, format="turtle")

    def query(self, sparql_query: str) -> List[Dict[str, Any]]:
        """Executes a SPARQL query and returns formatted dictionary results."""
        results = self.graph.query(sparql_query)
        output = []
        for row in results:
            row_dict = {}
            for var in results.vars:
                val = row[var]
                row_dict[str(var)] = val.toPython() if val is not None else None
            output.append(row_dict)
        return output

    def get_municipality_summary(self, cod_mun_6: str) -> List[Dict[str, Any]]:
        """Extracts all indicators for a specific municipality code."""
        clean_code = str(cod_mun_6).strip().zfill(6)[:6]

        query_str = f"""
        PREFIX sus: <http://example.org/sus/ontology#>
        PREFIX ibge: <http://example.org/ibge/ontology#>
        PREFIX cnes: <http://example.org/cnes/ontology#>
        PREFIX pbf: <http://example.org/pbf/ontology#>
        PREFIX inst_ibge: <http://example.org/ibge/municipality/>

        SELECT ?mun ?pop ?cases ?incidence ?facilities ?pbf_health ?pbf_prenatal
        WHERE {{
            BIND(inst_ibge:{clean_code} AS ?mun)
            
            ?snapshot sus:forMunicipality ?mun .
            
            OPTIONAL {{ ?snapshot ibge:totalPopulation ?pop . }}
            OPTIONAL {{ ?snapshot sus:totalCases ?cases . }}
            OPTIONAL {{ ?snapshot sus:incidenceRatePer100k ?incidence . }}
            OPTIONAL {{ ?snapshot cnes:facilityCount ?facilities . }}
            OPTIONAL {{ ?snapshot pbf:overallHealthComplianceRate ?pbf_health . }}
            OPTIONAL {{ ?snapshot pbf:prenatalComplianceRate ?pbf_prenatal . }}
        }}
        """
        return self.query(query_str)