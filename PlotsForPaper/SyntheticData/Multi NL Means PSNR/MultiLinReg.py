import pandas as pd
import statsmodels.formula.api as smf

dataset = pd.read_csv('OutputSignal3.txt')

model = smf.ols(formula = 'PSNR ~ Beta1+I(Beta1**2)', data = dataset).fit()
print model.summary()
print '\n\n\n'
model = smf.ols(formula = 'PSNR ~ T1+I(T1**2)', data = dataset).fit()
print model.summary()
print '\n\n\n'
model = smf.ols(formula = 'PSNR ~ Beta2+I(Beta2**2)', data = dataset).fit()
print model.summary()
print '\n\n\n'
model = smf.ols(formula = 'PSNR ~ T2+I(T2**2)', data = dataset).fit()
print model.summary()
print '\n\n\n'
model = smf.ols(formula = 'PSNR ~ Noise+Beta1+T1+Beta2+T2', data = dataset).fit()
print model.summary()
