import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from scipy.stats import norm
from pyts.approximation import SymbolicAggregateApproximation
from numpy.testing import rundocs

n_samples = 100 # liczba próbek: parametry do stworzenia przykładowego zestawu danych
n_timestamps = 24 # znaczniki czasowe

rgn = np.random.RandomState(41)
X = rgn.randn(n_samples, n_timestamps) # generowanie losowych danych

n_bins = 3 # liczba interwałów kwantyzacji
sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy='normal')
X_sax = sax.fit_transform(X)

# obliczanie interwałów kwantyzacji dla rozkładu Gaussa
# ppf = percent point function (odwrotna dystrybuanta)
bins = norm.ppf(np.linspace(0, 1, n_bins + 1)[1:-1])

bottom_bool = np.r_[True, X_sax[0, 1:] > X_sax[0, :-1]]

# wyświetlenie wykresu
plt.figure(figsize=(12, 8))
plt.plot(X[0], 'o--', label='Original')
for x, y, s, bottom in zip(range(n_timestamps), X[0], X_sax[0], bottom_bool):
    va = 'bottom' if bottom else 'top'
    plt.text(x, y, s, ha='center', verticalalignment=va, fontsize=24, color='#ff7f0e')
plt.hlines(bins, 0, n_timestamps, color='g', linestyles="--", linewidth=0.5)
sax_legend = mlines.Line2D([], [], color='#ff7f0e', marker='*', label='SAX - {0} bins'.format(n_bins))
first_legend = plt.legend(handles=[sax_legend], fontsize=14, loc=(0.79, 0.87))
ax = plt.gca().add_artist(first_legend)
plt.legend(loc=(0.835, 0.93), fontsize=14)
plt.show()
