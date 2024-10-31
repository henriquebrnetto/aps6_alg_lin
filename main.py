import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from utils import create_totals_df, erro, transform_array
import autograd.numpy as np_
from autograd import grad

folder = 'rice'
pattern = r'_(\d{4})\.'
N = 5 # Quantidade de anos usados para prever o n+1

df = create_totals_df(folder, pattern)


g = grad(erro)

x = np.array(df.index.astype(float))
y = df['yield'].to_numpy()

plt.scatter(x, y)
plt.xlabel('Ano')
plt.ylabel('Produção Mundial de Arroz (t/ha)')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

transformed_y = transform_array(y_train, N).T
transformed_x = transform_array(x_train, N).T

w_modelo = np.random.randn(N, 1)*400

alpha = 1e-10
b_modelo = 0.0

for _ in range(100): # Retirado do Notebook 6
    gradi = g((w_modelo, b_modelo, transformed_x, transformed_y))
    w_modelo -= alpha * gradi[0]
    b_modelo -= alpha * gradi[1]

trans_x_test, trans_y_test = transform_array(x_test, N), transform_array(y_test, N)