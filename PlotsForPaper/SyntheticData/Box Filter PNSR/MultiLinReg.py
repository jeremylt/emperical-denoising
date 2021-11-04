import pandas as pd
import statsmodels.formula.api as smf

dataset = pd.read_csv('OutputSignal3.txt')

model = smf.ols(formula = 'PSNR ~ Noise:Window', data = dataset).fit()

print model.summary()
