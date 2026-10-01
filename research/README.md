# Material histórico da iniciação científica

Esta pasta guarda os notebooks originais do projeto (2025-2026), como registro do caminho
percorrido. **Eles não são usados pelo app nem pelo pipeline atual.**

| Pasta | Conteúdo | Substituído por |
|---|---|---|
| `01_pretreatment/` | Modelo cinético do pré-tratamento com k por temperatura, geração LHS | `ethanol_ai/pretreatment.py`, `pipeline/01` e `03` |
| `02_enzymatic_hydrolysis/` | Modelo de Angarita et al. (2015) com parâmetros da literatura, geração LHS | `ethanol_ai/hydrolysis.py`, `pipeline/02` e `03` |
| `03_fermentation/` | Modelo de fermentação (Python e referência em MATLAB) | ainda não integrado |
| `04_genetic_ann_search/` | Busca de arquitetura de redes MLP por algoritmo genético | `ethanol_ai/surrogate.py`, `pipeline/04` |
| `05_back_optimization/` | Otimização reversa com a rede antiga | `ethanol_ai/optimization.py` e a aba Optimization do app |
| `legacy/` | Primeiros modelos (SVR, Random Forest, primeiras redes) | — |
| `legacy_artifacts/` | Dados sintéticos e redes `.h5` antigos, usados pelos notebooks acima | — |

Os caminhos dos notebooks foram atualizados para a estrutura atual do repositório, mas eles
não foram reexecutados.

Por que a abordagem mudou está documentado em [`docs/METODOLOGIA.md`](../docs/METODOLOGIA.md),
seção "O que mudou em relação à versão anterior".
