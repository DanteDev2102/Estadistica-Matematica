import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./src/datos_.csv")

# Mostrar las primeras filas del DataFrame
print("Primeras filas del DataFrame:")
print(df.head())

# Convertir variables categóricas a variables dummy
df = pd.get_dummies(df, drop_first=True)

# Calcular la matriz de correlaciones
correlation_matrix = df.corr()

# Mostrar la matriz de correlaciones
print("\nMatriz de Correlaciones:")
print(correlation_matrix)

# Visualizar la matriz de correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Matriz de Correlaciones")
plt.show()
