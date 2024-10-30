import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import create_totals_df

folder = 'data/rice'
pattern = r'_(\d{4})\.'

df = create_totals_df(folder, pattern)

x_train, x_test, y_train, y_test = train_test_split(df)

plt.plot(df)
plt.show()