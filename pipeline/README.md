# Pipeline: dos experimentos às redes do app

Cada etapa é um script independente, executado da raiz do repositório, nesta ordem.
Todas leem e escrevem em pastas fixas (`data/`, `models/`, `reports/`), então dá para
refazer só a partir da etapa que mudou.

```bash
pip install -r requirements-research.txt

python pipeline/00_build_experimental_data.py   # planilhas brutas -> data/experimental/*.csv
python pipeline/01_calibrate_pretreatment.py    # -> models/pretreatment_kinetics.json  (~20 min)
python pipeline/02_calibrate_hydrolysis.py      # -> models/hydrolysis_kinetics.json    (~8 min)
python pipeline/03_generate_synthetic.py        # -> data/synthetic/                    (~1 min)
python pipeline/04_train_surrogates.py          # -> models/*_surrogate.npz             (~1 h)
python pipeline/05_evaluate.py                  # -> reports/surrogate_evaluation.md
python -m pytest tests                          # confere física, fidelidade e o app
```

| Etapa | O que faz | Saída principal |
|---|---|---|
| 00 | Lê as planilhas originais, corrige erros e marca dados censurados | `data/experimental/pretreatment.csv`, `hydrolysis.csv` |
| 01 | Ajusta o modelo cinético do pré-tratamento (Arrhenius global + aquecimento) e estima a incerteza por bootstrap | `models/pretreatment_kinetics.json`, `reports/pretreatment_calibration.md` |
| 02 | Escolhe a estrutura do modelo de hidrólise por validação cruzada, calibra e faz bootstrap | `models/hydrolysis_kinetics.json`, `reports/hydrolysis_calibration.md` |
| 03 | Gera dados sintéticos com cada conjunto de parâmetros (ajuste central + bootstrap) | `data/synthetic/*.npz` (não versionado), `*_central.csv.gz` |
| 04 | Treina uma rede por conjunto de dados (ensemble) e exporta os pesos para NumPy | `models/pretreatment_surrogate.npz`, `hydrolysis_surrogate.npz` |
| 05 | Compara redes, modelo cinético e experimentos; confere as garantias físicas | `reports/surrogate_evaluation.md` |

## Quando refazer o quê

- **Novos experimentos de hidrólise ou pré-tratamento**: acrescente as linhas em
  `data/raw/` (ou adapte a etapa 00) e rode tudo a partir da 00.
- **Mudança no modelo cinético** (`ethanol_ai/pretreatment.py` ou `hydrolysis.py`): a partir da 01 ou 02.
- **Mudança nas faixas do app** (`ethanol_ai/domain.py`): se a faixa de treino mudar, a partir da 03.
- **Mudança na arquitetura das redes** (`ethanol_ai/surrogate.py` e `04_train_surrogates.py`):
  a partir da 04. As equações das "cabeças" físicas precisam ser iguais nos dois arquivos; a etapa
  04 confere isso ao final.

A metodologia completa está em [`docs/METODOLOGIA.md`](../docs/METODOLOGIA.md).
