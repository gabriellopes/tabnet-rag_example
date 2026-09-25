# app/src/kg/builder.py
import os
import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import XSD

# Definição dos Namespaces
SUS = Namespace("http://example.org/sus/ontology#")
IBGE = Namespace("http://example.org/ibge/ontology#")
PBF = Namespace("http://example.org/pbf/ontology#")
CNES = Namespace("http://example.org/cnes/ontology#")


class KnowledgeGraphBuilder:

    def __init__(self, ontology_path: str = "app/src/kg/ontology.ttl"):
        self.graph = Graph()
        self.ontology_path = ontology_path

        # Carrega a ontologia base se o arquivo existir
        if os.path.exists(self.ontology_path):
            self.graph.parse(self.ontology_path, format="turtle")

        # Registra prefixos no grafo
        self.graph.bind("sus", SUS)
        self.graph.bind("ibge", IBGE)
        self.graph.bind("pbf", PBF)
        self.graph.bind("cnes", CNES)

    def build_from_dataframe(self, df: pd.DataFrame) -> Graph:
        """Converte o DataFrame consolidado de municípios em triplas RDF."""
        for _, row in df.iterrows():
            cod_mun = str(row["cod_mun_6"]).strip()
            ano = int(row.get("ano", 2025))

            # URIs das Entidades
            mun_uri = URIRef(f"http://example.org/ibge/municipality/{cod_mun}")
            snapshot_uri = URIRef(
                f"http://example.org/sus/snapshot/{cod_mun}_{ano}"
            )

            # Instanciação do Município
            self.graph.add((mun_uri, RDF.type, IBGE.Municipality))
            self.graph.add(
                (mun_uri, IBGE.hasCode6, Literal(cod_mun, datatype=XSD.string))
            )

            # Instanciação do Snapshot Regional
            self.graph.add((snapshot_uri, RDF.type, SUS.HealthSnapshot))
            self.graph.add((snapshot_uri, SUS.forMunicipality, mun_uri))
            self.graph.add(
                (snapshot_uri, SUS.hasYear, Literal(ano, datatype=XSD.integer))
            )

            # Métricas IBGE
            if "pop" in row and pd.notna(row["pop"]):
                self.graph.add(
                    (
                        snapshot_uri,
                        IBGE.totalPopulation,
                        Literal(int(row["pop"]), datatype=XSD.integer),
                    )
                )

            # Métricas SINAN
            if "total_casos" in row and pd.notna(row["total_casos"]):
                self.graph.add(
                    (
                        snapshot_uri,
                        SUS.totalCases,
                        Literal(int(row["total_casos"]), datatype=XSD.integer),
                    )
                )
            if "taxa_incidencia" in row and pd.notna(row["taxa_incidencia"]):
                self.graph.add(
                    (
                        snapshot_uri,
                        SUS.incidenceRatePer100k,
                        Literal(
                            float(row["taxa_incidencia"]), datatype=XSD.float
                        ),
                    )
                )

            # Métricas CNES
            if "cnes_count" in row and pd.notna(row["cnes_count"]):
                self.graph.add(
                    (
                        snapshot_uri,
                        CNES.facilityCount,
                        Literal(int(row["cnes_count"]), datatype=XSD.integer),
                    )
                )

            # Métricas PBF / MDS
            if "perc_acomp_saude" in row and pd.notna(row["perc_acomp_saude"]):
                self.graph.add(
                    (
                        snapshot_uri,
                        PBF.overallHealthComplianceRate,
                        Literal(
                            float(row["perc_acomp_saude"]), datatype=XSD.float
                        ),
                    )
                )
            if "perc_gestantes_prenatal_em_dia" in row and pd.notna(
                row["perc_gestantes_prenatal_em_dia"]
            ):
                self.graph.add(
                    (
                        snapshot_uri,
                        PBF.prenatalComplianceRate,
                        Literal(
                            float(row["perc_gestantes_prenatal_em_dia"]),
                            datatype=XSD.float,
                        ),
                    )
                )

        return self.graph

    def export(
        self, output_path: str = "../data/processed/graph.ttl", fmt: str = "turtle"
    ):
        """Exporta o grafo resultante em disco."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.graph.serialize(destination=output_path, format=fmt)
        print(
            f"[KG Builder] Grafo exportado com sucesso para '{output_path}' ({len(self.graph)} triplas)."
        )