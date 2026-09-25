# app/src/rag/retriever.py
from typing import List, Dict, Any
from llama_index.core import Document, VectorStoreIndex
from app.src.rag.embeddings import EmbeddingModelLoader
from app.src.kg.sparql import SPARQLEngine

class GraphRAGRetriever:
    def __init__(self, graph_path: str = "data/processed/graph.ttl"):
        self.engine = SPARQLEngine(graph_path=graph_path)
        self.embed_model = EmbeddingModelLoader().load()
        self.index = None

    def build_index_from_kg(self):
        """Queries the KG snapshots and indexes them into LlamaIndex Documents."""
        query = """
        PREFIX sus: <http://example.org/sus/ontology#>
        PREFIX ibge: <http://example.org/ibge/ontology#>
        PREFIX cnes: <http://example.org/cnes/ontology#>
        PREFIX pbf: <http://example.org/pbf/ontology#>

        SELECT ?code ?pop ?cases ?incidence ?facilities ?pbf_health ?pbf_prenatal
        WHERE {
            ?mun a ibge:Municipality ; ibge:hasCode6 ?code .
            ?snapshot sus:forMunicipality ?mun ;
                      ibge:totalPopulation ?pop ;
                      sus:totalCases ?cases ;
                      sus:incidenceRatePer100k ?incidence ;
                      cnes:facilityCount ?facilities .
            OPTIONAL { ?snapshot pbf:overallHealthComplianceRate ?pbf_health . }
            OPTIONAL { ?snapshot pbf:prenatalComplianceRate ?pbf_prenatal . }
        }
        """
        records = self.engine.query(query)
        docs = []
        for r in records:
            content = (
                f"Municipality: {r.get('code')} | Population: {r.get('pop')} | "
                f"Disease Cases: {r.get('cases')} (Incidence: {r.get('incidence')}/100k) | "
                f"CNES Facilities: {r.get('facilities')} | "
                f"PBF Health Compliance: {r.get('pbf_health')}% | "
                f"PBF Prenatal Compliance: {r.get('pbf_prenatal')}%"
            )
            docs.append(Document(text=content, metadata={"cod_mun": r.get("code")}))

        self.index = VectorStoreIndex.from_documents(docs, embed_model=self.embed_model)

    def retrieve(self, query_str: str, top_k: int = 5) -> List[Any]:
        if not self.index:
            self.build_index_from_kg()
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query_str)