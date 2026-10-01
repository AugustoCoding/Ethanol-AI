"""
Fatores estequiométricos (massa de produto por massa de reagente).

Os modelos trabalham com massas "anidras" do polímero (glucana, xilana). As análises
medem monômeros e oligômeros como equivalentes de monômero (pós-hidrólise NREL) e os
furanos como tal; os fatores abaixo convertem entre as duas bases.
"""

R_GAS = 8.314462618e-3  # kJ/(mol·K)

# Hidrólise do polímero (anidro) a monômero: + H2O
GLUCAN_TO_GLUCOSE = 180.156 / 162.141      # 1.111
XYLAN_TO_XYLOSE = 150.130 / 132.115        # 1.136
GLUCAN_TO_CELLOBIOSE = 342.297 / 324.282   # 1.056 (celobiose por glucana)
CELLOBIOSE_TO_GLUCOSE = 2 * 180.156 / 342.297  # 1.053

# Desidratação do monômero a furano, expressa por massa de polímero anidro
HEXOSAN_TO_HMF = 126.110 / 162.141         # 0.778
PENTOSAN_TO_FURFURAL = 96.085 / 132.115    # 0.727
