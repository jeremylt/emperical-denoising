import pandas as pd
import statsmodels.formula.api as smf

dataset = pd.read_csv('OutputSignal3.txt')

model = smf.ols(formula = 'PSNR ~ Noise+Window+Beta*T+I(Window**2)+I(Beta**2)+I(T**2)', data = dataset).fit()

print model.summary()
