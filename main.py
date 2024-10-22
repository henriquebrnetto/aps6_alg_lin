import matplotlib.pyplot as plt
from utils import create_totals_df

folder = 'data/rice'
pattern = r'_(\d{4})\.'

df = create_totals_df(folder, pattern)


plt.plot(df)
plt.show()