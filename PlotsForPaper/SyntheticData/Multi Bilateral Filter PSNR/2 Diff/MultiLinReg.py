import pandas as pd
import statsmodels.formula.api as smf

dataset = pd.read_csv('2DiffOutputSignal3.txt')

model = smf.ols(formula = 'PSNR ~ SigD1+I(SigD1**2)', data = dataset).fit()
print model.summary()
print '\n\n\n'
model = smf.ols(formula = 'PSNR ~ SigI1+I(SigI1**2)', data = dataset).fit()
print model.summary()
print '\n\n\n'
model = smf.ols(formula = 'PSNR ~ SigD2+I(SigD2**2)', data = dataset).fit()
print model.summary()
print '\n\n\n'
model = smf.ols(formula = 'PSNR ~ SigI2+I(SigI2**2)', data = dataset).fit()
print model.summary()
print '\n\n\n'
model = smf.ols(formula = 'PSNR ~ Noise+SigD1+SigI1+SigD2+SigI2', data = dataset).fit()
print model.summary()
