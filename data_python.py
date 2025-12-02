# reproducer script (auto-saved)
# Input CSV: matrix_1000x200_sparse40.csv
# Rounding: nearest 0.5, clipped to [1,5]
# k = 202, lambda = 0.1, n_iters = 100
import numpy as np
import pandas as pd
R = pd.read_csv("matrix_1000x200_sparse40.csv", header=None).values.astype(float)
mask = (R != 0).astype(float)
# (Reproduce main ALS logic if needed.)
