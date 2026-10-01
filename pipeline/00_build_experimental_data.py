"""
Etapa 00: monta os conjuntos experimentais canônicos a partir dos arquivos brutos.

Entradas (data/raw/):
    pretreatment/experimental_data.csv   Tabelas 1-3 de Rocha et al. (2017), licor do pré-tratamento
    hydrolysis/OriginalSpreadsheet.xlsx  planilha original da hidrólise (duplicatas e controles)

Saídas (data/experimental/):
    pretreatment.csv   um ponto por temperatura x tempo, concentrações no licor (g/L)
    hydrolysis.csv     um ponto por condição x tempo, média corrigida e desvio-padrão (g/L)

Correções em relação ao antigo "Experimental Data.csv":
    - carga de enzima do ensaio 20% sólidos / 10 FPU: 0,233 g/L (o CSV repetia 0,175 g/L do ensaio 15%);
      a dose é por grama de celulose, então dobra junto com os sólidos (volume pipetado 0,325 vs 0,244 mL)
    - composição do sólido com os valores da planilha (65,55% celulose, 8,2% xilana)
    - celobiose "corrigida" igual a zero em t > 0 é censura (abaixo da celobiose do próprio extrato
      enzimático), não medida: fica marcada em cellobiose_censored
    - réplicas idênticas (desvio 0 em t > 0) ficam marcadas em replicates_identical
"""
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ethanol_ai.paths import RAW, EXPERIMENTAL  # noqa: E402

# Dose de enzima: g de proteína por FPU, obtida dos 7 ensaios consistentes
# (0,175 g/L para 10 FPU/g celulose, 150 g/L de sólidos e 66,55% de celulose)
G_PROTEIN_PER_FPU = 0.175 / (10 * 150 * 0.6655)

SHEETS = {
    # aba: (sólidos g/L, FPU/g celulose)
    "15%-5FPU": (150, 5),
    "15%-10FPU": (150, 10),
    "15FPU-15%": (150, 15),
    "20FPU-15%": (150, 20),
    "25FPU-15%": (150, 25),
    "30FPU-15%": (150, 30),
    "60FPU-15%": (150, 60),
    "20%-10FPU": (200, 10),
}
CELLULOSE_FRAC = 0.6655  # "Conc. Celulose" na planilha
XYLAN_FRAC = 0.082       # "Conc. Xilose palha (%)" na planilha
LIGNIN_FRAC = 0.252      # do antigo Processed Data.xlsx (não está na planilha original)


def build_pretreatment() -> pd.DataFrame:
    raw = pd.read_csv(RAW / "pretreatment" / "experimental_data.csv")
    raw.columns = raw.columns.str.strip()
    out = pd.DataFrame({
        "temperature_C": raw["Temperature [°C]"],
        "time_min": raw["Time [min]"],
        "solids_g_L": raw["Solid Loading [g/L]"],
        "cellulose_frac": raw["Cellulose Composition (%)"],
        "hemicellulose_frac": raw["Hemicellulose Composition (%)"],
        "lignin_frac": raw["Lignin Composition (%)"],
        "extractives_frac": raw["Extractives Composition (%)"],
        "ash_frac": raw["Ashes Composition (%)"],
        "glucose_g_L": raw["Glucose + Celobiose [g/L]"],
        "glucooligomers_g_L": raw["Glucooligomers [g/L]"],
        "hmf_g_L": raw["HMF [g/L]"],
        "formic_acid_g_L": raw["Formic acid [g/L]"],
        "xylose_g_L": raw["Xylose [g/L]"],
        "arabinose_g_L": raw["Arabinose [g/L]"],
        "xylooligomers_g_L": raw["Xylooligomers [g/L]"],
        "arabinooligomers_g_L": raw["Arabinooligomers [g/L]"],
        "furfural_g_L": raw["Furfural [g/L]"],
        "acetic_acid_g_L": raw["Acetic acid [g/L]"],
        "glucuronic_acid_g_L": raw["Glucuronic acid [g/L]"],
    })
    return out.sort_values(["temperature_C", "time_min"]).reset_index(drop=True)


def _parse_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Lê a segunda tabela da aba (valores já corrigidos pela diluição)."""
    start = df.index[df[0].astype(str).str.startswith("Corrigidas")][0]
    header = next(r for r in range(start, start + 6) if str(df.iat[r, 0]).strip().lower() == "tempo")
    rows = []
    r = header + 1
    while r < len(df) and pd.notna(df.iat[r, 0]) and not isinstance(df.iat[r, 0], str):
        rows.append({
            "time_h": float(df.iat[r, 0]),
            # Glicose: réplicas (1, 2), média corrigida pelo controle de enzima (4), desvio (5)
            "glucose_rep1": df.iat[r, 1], "glucose_rep2": df.iat[r, 2],
            "glucose_g_L": df.iat[r, 4], "glucose_sd": df.iat[r, 5],
            "xylose_rep1": df.iat[r, 11], "xylose_rep2": df.iat[r, 12],
            "xylose_g_L": df.iat[r, 14], "xylose_sd": df.iat[r, 15],
            "cellobiose_rep1": df.iat[r, 21], "cellobiose_rep2": df.iat[r, 22],
            "cellobiose_g_L": df.iat[r, 24], "cellobiose_sd": df.iat[r, 25],
        })
        r += 1
    return pd.DataFrame(rows)


def build_hydrolysis() -> pd.DataFrame:
    book = pd.ExcelFile(RAW / "hydrolysis" / "OriginalSpreadsheet.xlsx")
    parts = []
    for sheet, (solids, fpu) in SHEETS.items():
        d = _parse_sheet(book.parse(sheet, header=None))
        d.insert(0, "condition", f"S{solids}_E{fpu}FPU")
        d.insert(1, "solids_g_L", float(solids))
        d.insert(2, "enzyme_FPU_per_g_cellulose", float(fpu))
        d.insert(3, "enzyme_g_L", round(fpu * solids * CELLULOSE_FRAC * G_PROTEIN_PER_FPU, 4))
        d.insert(4, "cellulose_frac", CELLULOSE_FRAC)
        d.insert(5, "hemicellulose_frac", XYLAN_FRAC)
        d.insert(6, "lignin_frac", LIGNIN_FRAC)
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    num = [c for c in out.columns if c.endswith(("_g_L", "_sd")) or "_rep" in c]
    out[num] = out[num].astype(float).round(4)
    t = out["time_h"] > 0
    out["cellobiose_censored"] = t & (out["cellobiose_g_L"] <= 0)
    out["replicates_identical"] = t & (out["glucose_rep1"] == out["glucose_rep2"])
    return out


def main() -> None:
    EXPERIMENTAL.mkdir(parents=True, exist_ok=True)
    pre = build_pretreatment()
    hyd = build_hydrolysis()
    pre.to_csv(EXPERIMENTAL / "pretreatment.csv", index=False)
    hyd.to_csv(EXPERIMENTAL / "hydrolysis.csv", index=False)

    print(f"pretreatment.csv: {len(pre)} pontos, temperaturas {sorted(pre.temperature_C.unique())}")
    print(f"hydrolysis.csv: {len(hyd)} pontos em {hyd.condition.nunique()} condições")
    print(hyd.groupby("condition")[["solids_g_L", "enzyme_g_L"]].first().to_string())
    print(f"celobiose censurada: {int(hyd.cellobiose_censored.sum())} pontos; "
          f"réplicas idênticas: {int(hyd.replicates_identical.sum())} pontos")

    # Conferência contra o CSV antigo: só a carga de enzima do ensaio 20% deve divergir
    old = pd.read_csv(RAW / "hydrolysis" / "Experimental Data.csv")
    old.columns = old.columns.str.strip()
    old = old[old["Time [h]"] > 0].assign(key=lambda d: d["Glucose Concentration [g/L]"].round(1))
    new = hyd[hyd.time_h > 0].assign(key=lambda d: d["glucose_g_L"].round(1))
    merged = new.merge(old, left_on=["solids_g_L", "time_h", "key"],
                       right_on=["Solids Loading [g/L]", "Time [h]", "key"], how="left")
    assert merged["Enzyme Loading [g/L]"].notna().all(), "pontos sem correspondência no CSV antigo"
    diff = merged[(merged["Enzyme Loading [g/L]"] - merged["enzyme_g_L"]).abs() > 0.002]
    print("condições com carga de enzima diferente do CSV antigo:",
          sorted(diff.condition.unique()), f"(CSV antigo: {sorted(diff['Enzyme Loading [g/L]'].unique())} g/L)")


if __name__ == "__main__":
    main()
