import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import create_totals_df, erro, transform_array
import autograd.numpy as np_
from autograd import grad

folder = 'rice'
pattern = r'_(\d{4})\.'
N = 5 # Quantidade de anos usados para prever o n+1

df = create_totals_df(folder, pattern)


g = grad(erro)

x = df.index.astype(float).to_numpy()
y = df['yield'].to_numpy()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

transformed_x, transformed_y = transform_array(x_train, N).T, transform_array(y_train, N).T

w_modelo = np.random.randn(N, 1)
alpha = 1e-10
b_modelo = 0.0

erros = {
    'a' : [w_modelo],
    'erro' : [erro( (w_modelo, b_modelo, transformed_x, transformed_y) )]
}

for _ in range(1000): # Retirado do Notebook 6
    gradi = g((w_modelo, b_modelo, transformed_x, transformed_y))
    w_modelo -= alpha * gradi[0]
    b_modelo -= alpha * gradi[1]

    erros['a'] += [w_modelo]
    erros['erro'] += [erro( (w_modelo, b_modelo, transformed_x, transformed_y) )]

erros_df = pd.DataFrame(erros)
min_error = erros_df.loc[erros_df['erro'] == min(erros_df['erro']), 'a'].values[0] # Seleciona pesos de acordo com o menor erro encontrado

trans_x_test, trans_y_test = transform_array(x_test, N), transform_array(y_test, N)

pred_y = trans_x_test @ min_error # Calcula previsão dos novos valores


# Gráfico comparando valores de treino, teste e previsão
plt.scatter(x_train, y_train, color='red', label='Train Data')
plt.scatter(x_test[N-1:], pred_y, color='black', label='Test Data (Prediction)')
plt.scatter(x_test, y_test, color='blue', label='Test Data (Prediction)')
plt.xlabel('Ano')
plt.ylabel('Produção Mundial de Arroz (t/ha)')
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()
