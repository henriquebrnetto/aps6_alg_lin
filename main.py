import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from utils import create_totals_df, erro
import autograd.numpy as np_
from autograd import grad

folder = 'rice'
pattern = r'_(\d{4})\.'

df = create_totals_df(folder, pattern)

g = grad(erro)

x = np.array(df.index.astype(float))
y = df['yield'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)
                                                    
a_modelo = 400.0
alpha = 0.0001

for _ in range(10):
    gradi = g((a_modelo, x, y))
    a_modelo -= alpha * gradi[0]

# plt.plot(df)
# plt.show()