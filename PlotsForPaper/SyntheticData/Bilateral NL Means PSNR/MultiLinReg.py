import pandas as pd
import statsmodels.formula.api as smf

dataset = pd.read_csv('OutputSignal3.txt')

model = smf.ols(formula = 'PSNR ~ SigD+I(SigD**2)', data = dataset).fit()
print model.summary()
print '\n\n\n'
model = smf.ols(formula = 'PSNR ~ SigI+I(SigI**2)', data = dataset).fit()
print model.summary()
print '\n\n\n'
model = smf.ols(formula = 'PSNR ~ Beta+I(Beta**2)', data = dataset).fit()
print model.summary()
print '\n\n\n'
model = smf.ols(formula = 'PSNR ~ T+I(T**2)', data = dataset).fit()
print model.summary()
print '\n\n\n'
model = smf.ols(formula = 'PSNR ~ Noise+SigD+SigI+I(Beta**2)', data = dataset).fit()
print model.summary()
