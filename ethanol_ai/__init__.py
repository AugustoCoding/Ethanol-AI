"""
Ethanol AI: modelos do processo de etanol de segunda geração a partir de palha de cana.

Fluxo do projeto (ver docs/METODOLOGIA.md):
    experimentos -> modelos cinéticos calibrados -> dados sintéticos -> redes neurais -> app

Módulos:
    paths          caminhos do repositório
    constants      fatores estequiométricos
    experimental   leitura dos dados experimentais canônicos
    domain         faixas de operação (experimentais, de geração de dados e do app)
    pretreatment   modelo cinético do pré-tratamento hidrotérmico (Arrhenius + aquecimento)
    hydrolysis     modelo cinético da hidrólise enzimática (Angarita et al., 2015)
    surrogate      redes neurais com estrutura física (inferência em NumPy)
    optimization   busca de condições de operação a partir das redes
"""
