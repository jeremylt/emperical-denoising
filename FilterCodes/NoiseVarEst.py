# Note: This file can't run on its own.

# Estimate Noise Variance
varest = 0

for i in range(1, DataLen - 1):
    varest += (0.5*Data[i - 1] - Data[i] + 0.5*Data[i + 1])**2.0

varest *= 2./(3.*(DataLen - 2))
