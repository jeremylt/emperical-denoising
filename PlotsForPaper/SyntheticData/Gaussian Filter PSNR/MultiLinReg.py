import pandas as pd
import statsmodels.formula.api as smf

dataset = pd.read_csv('OutputSignal3.txt')

model = smf.ols(formula = 'PSNR ~ Noise*SigD+I(SigD**2)', data = dataset).fit()

print model.summary()
