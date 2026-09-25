# app/src/etl/aggregator.py
import pandas as pd
from app.src.etl.readers.dbc import DBCReader
from app.src.etl.readers.dbf import DBFReader
from app.src.etl.readers.csv import CSVReader

def clean_code(series: pd.Series) -> pd.Series:
    """Normalizes IBGE municipality codes to standard 6-digit strings."""
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(6)
        .str[:6]
    )

class DatasetAggregator:
    def __init__(self, sinan_path: str, ibge_path: str, cnes_path: str, pbf_path: str):
        self.sinan_path = sinan_path
        self.ibge_path = ibge_path
        self.cnes_path = cnes_path
        self.pbf_path = pbf_path

    def load_and_aggregate(self) -> pd.DataFrame:
        # 1. Load Data
        df_sinan = DBCReader(self.sinan_path).load_data()
        df_ibge  = DBFReader(self.ibge_path).load_data()
        df_cnes  = DBCReader(self.cnes_path).load_data()
        df_pbf   = CSVReader(self.pbf_path).load_data()

        # 2. Normalize IBGE Codes
        cnes_col = "CODUFMUN" if "CODUFMUN" in df_cnes.columns else ("MUNICIP" if "MUNICIP" in df_cnes.columns else "COD_MUN")

        df_sinan["cod_mun_6"] = clean_code(df_sinan["ID_MN_RESI"])
        df_ibge["cod_mun_6"]  = clean_code(df_ibge["cod_mun"])
        df_cnes["cod_mun_6"]  = clean_code(df_cnes[cnes_col])
        df_pbf["cod_mun_6"]   = clean_code(df_pbf["codigo_ibge"])

        # 3. Aggregate Individual Datasets
        # SINAN: Disease case count
        sinan_agg = df_sinan.groupby("cod_mun_6").size().reset_index(name="total_casos")

        # IBGE: Population baseline
        ibge_agg = df_ibge.groupby("cod_mun_6")["pop"].sum().reset_index() if "pop" in df_ibge.columns else df_ibge[["cod_mun_6"]].drop_duplicates()

        # CNES: Unique health facility count
        cnes_col_id = "CNES" if "CNES" in df_cnes.columns else cnes_col
        cnes_agg = df_cnes.groupby("cod_mun_6")[cnes_col_id].nunique().reset_index(name="cnes_count")

        # MDS / PBF: Average welfare health compliance rates
        pbf_cols = [c for c in ["perc_acomp_saude", "perc_gestantes_prenatal_em_dia"] if c in df_pbf.columns]
        pbf_agg = df_pbf.groupby("cod_mun_6")[pbf_cols].mean().reset_index()

        # 4. Consolidate via Inner Join (Intersection)
        merged = (
            ibge_agg
            .merge(sinan_agg, on="cod_mun_6", how="inner")
            .merge(cnes_agg, on="cod_mun_6", how="inner")
            .merge(pbf_agg, on="cod_mun_6", how="inner")
        )

        # 5. Calculate Derived Health Metrics
        if "pop" in merged.columns and "total_casos" in merged.columns:
            merged["taxa_incidencia"] = (merged["total_casos"] / merged["pop"]) * 100000
            merged["taxa_incidencia"] = merged["taxa_incidencia"].round(2)

        merged["ano"] = 2025
        return merged