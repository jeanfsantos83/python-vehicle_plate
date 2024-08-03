import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carregar a planilha de amostra
df = pd.read_excel('amostra_geotecnica.xlsx')

# Verificar se a planilha foi carregada corretamente
print(df.head())

# Análise de dados
# 1. Estatística descritiva
print("Estatística descritiva:")
print(df.describe())

# 2. Distribuição de frequência
print("\nDistribuição de frequência:")
print(df['tipo_de_solo'].value_counts())

# 3. Correlação entre variáveis
print("\nCorrelação entre variáveis:")
corr_matrix = df.corr()
print(corr_matrix)

# 4. Gráficos
# 4.1. Histograma de densidade
plt.hist(df['densidade'], bins=50)
plt.xlabel('Densidade')
plt.ylabel('Frequência')
plt.title('Histograma de densidade')
plt.show()

# 4.2. Boxplot de resistência
plt.boxplot(df['resistencia'])
plt.xlabel('Resistência')
plt.ylabel('Valor')
plt.title('Boxplot de resistência')
plt.show()

# 5. Análise de outliers
print("\nAnálise de outliers:")
Q1 = df['resistencia'].quantile(0.25)
Q3 = df['resistencia'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[~((df['resistencia'] >= Q1 - 1.5 * IQR) & (df['resistencia'] <= Q3 + 1.5 * IQR))]
print(outliers)

# 6. Agrupamento de dados
print("\nAgrupamento de dados:")
grouped_df = df.groupby('tipo_de_solo')
print(grouped_df.mean())

# 7. Regressão linear
print("\nRegressão linear:")
from sklearn.linear_model import LinearRegression
X = df[['densidade']]
y = df['resistencia']
model = LinearRegression()
model.fit(X, y)
print("Coeficiente de determinação (R²):", model.score(X, y))
print("Coeficiente de inclinação:", model.coef_)
print("Intercepto:", model.intercept_)
