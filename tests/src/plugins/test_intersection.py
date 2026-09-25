# tests/src/plugins/test_intersection.py
import os
import pytest
import pandas as pd
from app.src.readers.dbc import DBCReader
from app.src.readers.dbf import DBFReader
from app.src.readers.csv import CSVReader

SINAN_FILE = "data/raw/sus/sinan/cancer_trab/CANCBR25.dbc"
IBGE_FILE  = "data/raw/sus/ibge/POPSBR25/POP25.dbf"
CNES_FILE  = "data/raw/sus/cnes/STBA2512.dbc"
PBF_FILE   = "data/raw/mds/pbf/2025.csv"

def clean_code(series: pd.Series) -> pd.Series:
    """Normalizes IBGE municipality codes to standard 6-digit strings."""
    return (
        series.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(6)
        .str[:6]
    )

@pytest.mark.skipif(
    not (os.path.exists(SINAN_FILE) and os.path.exists(IBGE_FILE) and os.path.exists(CNES_FILE) and os.path.exists(PBF_FILE)),
    reason="One or more raw data files missing locally"
)
def test_four_dataset_municipality_intersection():
    df_sinan = DBCReader(SINAN_FILE).load_data()
    df_ibge  = DBFReader(IBGE_FILE).load_data()
    df_cnes  = DBCReader(CNES_FILE).load_data()
    df_pbf   = CSVReader(PBF_FILE).load_data()

    cnes_col = "CODUFMUN" if "CODUFMUN" in df_cnes.columns else ("MUNICIP" if "MUNICIP" in df_cnes.columns else "COD_MUN")

    # Add normalized 6-digit IBGE column to each DataFrame
    df_sinan["cod_mun_6"] = clean_code(df_sinan["ID_MN_RESI"])
    df_ibge["cod_mun_6"]  = clean_code(df_ibge["cod_mun"])
    df_cnes["cod_mun_6"]  = clean_code(df_cnes[cnes_col])
    df_pbf["cod_mun_6"]   = clean_code(df_pbf["codigo_ibge"])

    # Compute intersection set
    muns_sinan = set(df_sinan["cod_mun_6"].dropna())
    muns_ibge  = set(df_ibge["cod_mun_6"].dropna())
    muns_cnes  = set(df_cnes["cod_mun_6"].dropna())
    muns_pbf   = set(df_pbf["cod_mun_6"].dropna())

    intersection = sorted(list(muns_sinan & muns_ibge & muns_cnes & muns_pbf))

    print(f"\n=== MUTUAL MUNICIPALITIES ({len(intersection)}) ===")
    print(f"IBGE 6-Digit Codes: {intersection}")

    print("\n=== RELEVANT VARIABLES AVAILABLE FOR FEATURE MATRIX ===")
    print(f"[1] SINAN Disease/Demographics : ['cod_mun_6', 'ID_AGRAVO', 'CS_RACA', 'CS_SEXO', 'NU_IDADE_N']")
    print(f"[2] IBGE Population Baseline   : {list(df_ibge.columns)}")
    print(f"[3] CNES Facility Infrastructure: {[c for c in ['cod_mun_6', 'CNES', 'TP_UNIDADE', 'NIV_DEP'] if c in df_cnes.columns or c == 'cod_mun_6']}")
    print(f"[4] MDS/PBF Welfare Compliance : {[c for c in ['cod_mun_6', 'perc_acomp_saude', 'perc_gestantes_prenatal_em_dia', 'perc_criancas_cumpriram_saude'] if c in df_pbf.columns or c == 'cod_mun_6']}")

    assert len(intersection) > 0