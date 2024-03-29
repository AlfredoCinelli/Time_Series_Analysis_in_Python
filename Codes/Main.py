"""

@Author: Alfredo Cinelli

Introduction to Time Series Analysis in Python

"""

# %% Import the relevant packages

from Johansen import coint_johansen  # dedicated function
from Granger import granger  # dedicated function
import pmdarima as pm
from scipy.stats import kstest
from arch.unitroot import ADF
from arch import arch_model
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 7)

# %% Import and prepare data

Data_msft = yf.download(
    'MSFT',
    start='2004-01-02',
    end='2021-04-30',
    progress=False)  # import data for Microsoft
Data_intc = yf.download(
    'INTC',
    start='2004-01-02',
    end='2021-04-30',
    progress=False)  # import data for Intel
cp_msft = Data_msft['Close']  # closing prices for Microsoft
cp_msft.name = 'MSFT'  # rename the Series
cp_intc = Data_intc['Close']  # closing prices for Intel
cp_intc.name = 'INTC'  # rename the Series
# summary statistics for stocks prices
sum_stat = pd.concat([cp_msft.describe(), cp_intc.describe()], axis=1)
print(sum_stat)  # display


'''--------------------------- PRICE ANALYSIS ------------------------------'''

# %% Data visualization

pd.concat((cp_msft, cp_intc), axis=1).plot(
    grid=True, ylabel='$P_t$', title='STOCK CLOSING PRICES')  # prices plot
plt.savefig(path + 'cp_plot' + '.pdf', bbox_inches='tight')
fig, axes = plt.subplots(2, 1, sharex=True)  # create a subplot
sm.graphics.tsa.plot_acf(cp_msft, lags=30, title='ACF MSFT PRICES', ax=axes[0])
sm.graphics.tsa.plot_acf(
    cp_intc,
    lags=30,
    title='ACF INTEL PRICES',
    ax=axes[1])
# plt.savefig(path+'cp_ACF'+'.pdf',bbox_inches='tight')

# %% Unit root test (Augmented Dickey-Fueller)

# H0: Unit-root vs H1: Stationary (weekly)
# Augmented DF test with BIC as criterion for lag length
print(ADF(cp_msft, trend='ct', method='BIC'))
# Augmented DF test with BIC as criterion for lag length
print(ADF(cp_intc, trend='c', method='BIC'))


''' --------------------------------- COMMENT --------------------------------
From the time series plot it can be seen that the two price series are not stationary
and seem to show a trend over time. Moreover as it can be seen from the Correlogram
for the two series it does not decay to 0 remaining more or less constant over lags
showing a lack of statinarity.
All the above considerations are then "formally" corroborated by the Augmented Dickey-
Fuller test. In the test lags are taken into consideration to account for the
autocorrelation in the reisiduals of the auxiliary regression. The lag lenght of the
test is based on the BIC information criterion. The Null is that the series is Unit-root
indeed both the prices are actually non-stationary. More in detail for the Microsoft
prices 26 lags have been used and for Intel prices 9 lags have been used, finally for
the Microsoft series a constant and linear time trend has been added because the price
plot seems to show that, whilst for the Intel series just a constant drift has been
considered because its plot seems to show no linear time trend. '''


'''--------------------------- RETURN ANALYSIS ------------------------------'''

# %% Data preparation and visualization

r_msft = np.log(cp_msft).diff().dropna()  # log-returns for Microsoft
r_intc = np.log(cp_intc).diff().dropna()  # log-returns for Intel
pd.concat((r_msft, r_intc), axis=1).plot(
    grid=True, ylabel='$r_t$', title='STOCK LOG-RETURNS')
plt.savefig(path + 'r_plot' + '.pdf', bbox_inches='tight')
plt.show()
plt.close()
fig, axes = plt.subplots(2, 2, sharex=True)
sm.graphics.tsa.plot_acf(
    r_msft, lags=10, title='MICROSOFT LOG-RETURNS ACF', ax=axes[0, 0])
sm.graphics.tsa.plot_pacf(
    r_msft, lags=10, title='MICROSOFT LOG-RETURNS PACF', ax=axes[0, 1])
sm.graphics.tsa.plot_acf(
    r_intc, lags=10, title='INTEL LOG-RETURNS ACF', ax=axes[1, 0])
sm.graphics.tsa.plot_pacf(
    r_intc, lags=10, title='INTEL LOG-RETURNS PACF', ax=axes[1, 1])
plt.show()
# fig.savefig(path+'r_ACF_PACF'+'.pdf',bbox_inches='tight')
plt.close()

# %% Stylized facts of returns

fig, axes = plt.subplots(2, 1, sharex=True)
sm.qqplot(r_msft, line='s', fit=True, ax=axes[0])
sm.qqplot(r_intc, line='s', ax=axes[1])
plt.show()
# fig.savefig(path+'r_qq_plot'+'.pdf',bbox_inches='tight')
plt.close()
# Kolmogorov-Smirnov test H0: Gaussian distribution vs H1: Not Gaussian
print('Kolmogorov-Smirnov test p-value: ', kstest(r_msft, 'norm')[1])
# Kolmogorov-Smirnov test H0: Gaussian distribution vs H1: Not Gaussian
print('Kolmogorov-Smirnov test p-value: ', kstest(r_intc, 'norm')[1])
# Jarque-Berta test H0: Gaussian Distribution vs H1: Not Gaussia
print('Jarque-Bera test p-value: ', sm.stats.stattools.jarque_bera(r_msft)[1])
# Jarque-Berta test H0: Gaussian Distribution vs H1: Not Gaussia
print('Jarque-Bera test p-value: ', sm.stats.stattools.jarque_bera(r_intc)[1])
fig, axes = plt.subplots(2, 1, sharex=True)
sm.graphics.tsa.plot_acf(
    r_msft**2,
    lags=15,
    title='MICROSOFT SQUARED LOG-RETURNS ACF',
    ax=axes[0])
sm.graphics.tsa.plot_acf(
    r_intc**2,
    lags=15,
    title='INTEL SQUARED LOG-RETURNS ACF',
    ax=axes[1])
plt.show()
# fig.savefig(path+'r2_ACF'+'.pdf',bbox_inches='tight')
plt.close()
# Augmented DF test with BIC as criterion for lag length
print(ADF(r_msft, trend='n', method='BIC'))
# Augmented DF test with BIC as criterion for lag length
print(ADF(r_intc, trend='n', method='BIC'))
print(
    'ARCH test p-value: ',
    np.around(
        sm.stats.diagnostic.het_arch(r_msft)[1],
        2))  # ARCH test H0: No ARCH effects vs H1: ARCH effects
print(
    'ARCH test p-value: ',
    np.around(
        sm.stats.diagnostic.het_arch(r_intc)[1],
        2))  # ARCH test H0: No ARCH effects vs H1: ARCH effects
print(
    'Ljung-Box test p-value: ',
    np.around(
        sm.stats.diagnostic.acorr_ljungbox(
            r_msft,
            lags=10)[1],
        2))  # Ljung-Box test H0: No autocorrelation vs H1: autocorrelation
print(
    'Ljung-Box test p-value: ',
    np.around(
        sm.stats.diagnostic.acorr_ljungbox(
            r_intc,
            lags=10)[1],
        2))  # Ljung-Box test H0: No autocorrelation vs H1: autocorrelation

''' --------------------------------- COMMENT --------------------------------
Several stylized facts of log-returns have been investigated: First from the QQ plot
it can be seen that log-returns are not Gauissan because of fat tails. The latter
fact has been confirmed by running two tests, the Kolmogorov-Smirnov test for Gaussianity
and the Jarque-Bera test and both the Null of the tests have been rejected. Moreover
log-returns are stationary as it can be seen from the correlograms and as confirmed
by the Augmented Dickey-Fueller test. Moreover, it worth noting how squared returns
show a far greater (positive) autocorrelation over time compared to plain returns.
Moreover by running the Engle ARCH test, the feeling of conditional heteroskedasticity in
log-returns has been confirmed.'''

# %% ARMA modelling

# pm.auto_arima(r_msft,max_p=5,max_q=5,information_criterion='bic',test='adf') # --> ARIMA(0,0,1)
# pm.auto_arima(r_intc,max_p=5,max_q=5,information_criterion='bic',test='adf')
# # --> ARIMA(2,0,2)
# initialize the model for Microsoft log-returns
mdl_msft = sm.tsa.ARIMA(r_msft, order=(0, 0, 1))
res_msft = mdl_msft.fit(method='mle')  # fit the model via Maximum Likelihood
# as it can be seen all the parameters are statistically significant
print(res_msft.summary())
# take the model standardized residuals
e_msft = res_msft.resid / (res_msft.sigma2**0.5)
# initialize the model for the Intel log-returns
mdl_intc = sm.tsa.ARIMA(r_intc, order=(2, 0, 2))
# fit the model via Maximum Likelihood
res_intc = mdl_intc.fit(method='mle', trend='nc')
# as it can be seen all the parameters are statistically significant
print(res_intc.summary())
# take the model standardized residuals
e_intc = res_intc.resid / (res_intc.sigma2**0.5)
e_msft.name = 'Microsoft'
e_intc.name = 'Intel'

''' --------------------------------- COMMENT --------------------------------
The ARIMA modelling has been applied to the two stocks, the optimal lag length has
been estimated using Information Criterion. More specifically the models lag length
is the one that minimizes the BIC criterion. The choice of the BIC over other IC as
the AIC is due to its parsimonius behaviour. Indeed, in my opinion, a parsimonious
model is preferable for several reasons: the RSS is inversely proportional to the
number of degrees of freedom, complex models usually do well in sample but not out
of sample because they tend to capture not only the signal but also the noise. Moreover
all the parameters estimated in the two models are statistically significant, for the
Intel returns the constant has been suppressed. '''

# %% Diagnostic cheking

pd.concat(
    (e_msft,
     e_intc),
    axis=1).plot(
        grid=True,
        ylabel='$\\epsilon_t$',
    title='ARIMA MODELS STANDARDIZED RESIDUALS')
# plt.savefig(path+'ARMA_e'+'.pdf',bbox_inches='tight')
plt.show()
plt.close()
print(
    'Ljung-Box test p-value: ',
    np.around(
        sm.stats.diagnostic.acorr_ljungbox(
            e_msft,
            lags=10)[1],
        2))  # Ljung-Box test H0: No autocorrelation vs H1: autocorrelation
print(
    'Ljung-Box test p-value: ',
    np.around(
        sm.stats.diagnostic.acorr_ljungbox(
            e_intc,
            lags=10)[1],
        2))  # Ljung-Box test H0: No autocorrelation vs H1: autocorrelation
print(
    'ARCH test p-value: ',
    np.around(
        sm.stats.diagnostic.het_arch(e_msft)[1],
        2))  # ARCH test H0: No ARCH effects vs H1: ARCH effects
print(
    'ARCH test p-value: ',
    np.around(
        sm.stats.diagnostic.het_arch(e_intc)[1],
        2))  # ARCH test H0: No ARCH effects vs H1: ARCH effects
fig, axes = plt.subplots(2, 2, sharex=True)
sm.graphics.tsa.plot_acf(
    e_msft, lags=15, title='MICROSOFT ARIMA STD. RESIDUALS ACF', ax=axes[0, 0])
sm.graphics.tsa.plot_pacf(
    e_intc, lags=15, title='MICROSOFT ARIMA STD. RESIDUALS PACF', ax=axes[0, 1])
sm.graphics.tsa.plot_acf(
    e_msft, lags=15, title='INTEL ARIMA STD. RESIDUALS ACF', ax=axes[1, 0])
sm.graphics.tsa.plot_pacf(
    e_intc, lags=15, title='INTEL ARIMA STD. RESIDUALS PACF', ax=axes[1, 1])
plt.show()
# fig.savefig(path+'ARMA_e_ACF_PACF'+'.pdf',bbox_inches='tight')
plt.close()

''' --------------------------------- COMMENT --------------------------------
From the residuals diagnostic it can be seen that for the Microsoft standardized residuals there
is some autocorrelation from lag 4 onward, but at the end looking to ACF no serious autocorrelation is there.
Whilist for the Intel standardized residuals there is no autocorrelation over time. The lack of autocorrelation
is a desirable outcome of the ARIMA model residuals. Moreover also the correlogram are
showing no autocorrelation in the residuals of the two ARIMA models.
Finally the presence of ARCH effect, thus heteroskedasticity, has been investigated.
Indedd as the theory suggests and also the clusters in the residuals plot, both the
ARCH tests are corroborating the presence of heteroskedasticity in the models residuals'''

'''--------------------------- VOLATILITY ANALYSIS ------------------------------'''

# %% GARCH modelling (better calibrating the model on % returns)

# models are Zero mean because are calibrated on residuals of ARIMA (i.e.
# demeaned log-returns)
e_msft = res_msft.resid  # take residuals from ARIMA model
e_intc = res_intc.resid  # take residuals from ARIMA model
# initialize a GJR(1,1) on ARIMA residual with Student distribution
mdl_msft = arch_model(
    e_msft * 100,
    mean='Zero',
    p=1,
    q=1,
    o=1,
    dist='StudentsT')
res_msft = mdl_msft.fit(disp='off')  # fit the model via Maximum Likelihood
# all the parameters are significant thus there is asimmetry  and nu is
# small confirming lack of Gaussianity and lambda is not significant thus
# no skweness
print(res_msft.summary())
# initialize a GJR(1,1) on ARIMA residual with Student distribution
mdl_intc = arch_model(e_intc * 100, mean='Zero', p=1, q=1, o=1, dist='skewt')
res_intc = mdl_intc.fit(disp='off')  # fit the model via Maximum Likelihood
# gamma is not significant thus no asimmetry and nu is small thus
# confirming lack of Gaussianity and lambda is significant thus skewness
print(res_intc.summary())
# initialize a GARCH(1,1) on ARIMA residual with Student distribution
mdl_intc = arch_model(e_intc * 100, mean='Zero', p=1, q=1, dist='skewt')
res_intc = mdl_intc.fit(disp='off')  # fit the model via Maximum Likelihood
# all the parameters are singificant and nu is small thus confirming lack
# of Gaussianity
print(res_intc.summary())
# get estimated conditional volatility
s_msft = res_msft.conditional_volatility
# get estimated conditional volatlility
s_intc = res_intc.conditional_volatility
z_msft = res_msft.std_resid  # take the standardized residuals of the model
z_intc = res_intc.std_resid  # take the standardized residuals of the model

# %% Check the GARCH models

s_msft.name = 'Microsoft'
s_intc.name = 'Intel'
z_msft.name = 'Microsoft'
z_intc.name = 'Intel'
pd.concat(
    (s_msft,
     s_intc),
    axis=1).plot(
        grid=True,
        ylabel='$\\sigma_t$',
    title='PERCENTAGE CONDITIONAL VOLATILITY')
plt.savefig(path + 'GARCH_s' + '.pdf', bbox_inches='tight')
plt.show()
plt.close()
pd.concat((z_msft, z_intc), axis=1).plot(
    grid=True, ylabel='$z_t$', title='STANDARDIZED RESIDUALS')
# plt.savefig(path+'GARCH_z'+'.pdf',bbox_inches='tight')
plt.show()
plt.close()
fig, axes = plt.subplots(2, 2, sharex=True)
sm.graphics.tsa.plot_acf(
    z_msft, lags=15, title='MICROSOFT GARCH STD. RESIDUALS ACF', ax=axes[0, 0])
sm.graphics.tsa.plot_pacf(
    z_intc, lags=15, title='MICROSOFT GARCH STD. RESIDUALS PACF', ax=axes[0, 1])
sm.graphics.tsa.plot_acf(
    z_msft, lags=15, title='INTEL GARCH STD. RESIDUALS ACF', ax=axes[1, 0])
sm.graphics.tsa.plot_pacf(
    z_intc, lags=15, title='INTEL GARCH STD. RESIDUALS PACF', ax=axes[1, 1])
plt.show()
# fig.savefig(path+'z_ACF_PACF'+'.pdf',bbox_inches='tight')
plt.close()
print(
    'Ljung-Box test p-value: ',
    np.around(
        sm.stats.diagnostic.acorr_ljungbox(
            z_msft,
            lags=10)[1],
        2))  # Ljung-Box test H0: No autocorrelation vs H1: autocorrelation
print(
    'Ljung-Box test p-value: ',
    np.around(
        sm.stats.diagnostic.acorr_ljungbox(
            z_intc,
            lags=10)[1],
        2))  # Ljung-Box test H0: No autocorrelation vs H1: autocorrelation
print(
    'ARCH test p-value: ',
    np.around(
        sm.stats.diagnostic.het_arch(z_msft)[1],
        2))  # ARCH test H0: No ARCH effects vs H1: ARCH effects
print(
    'ARCH test p-value: ',
    np.around(
        sm.stats.diagnostic.het_arch(z_intc)[1],
        2))  # ARCH test H0: No ARCH effects vs H1: ARCH effects

''' --------------------------------- COMMENT --------------------------------
GARCH has been calibrated on ARIMA residuals, thus on returns already demeaned of conditinal mean.
The GARCH model are a GJR(1,1) with Student distribution for Microsoft, because the gamma parameter is
both significant and positive as it has to be. Whilist for the Intel the gamma parameter is not singificant
and the lambda parameter, unlike Microsoft, is significant thus a skew-t GARCH(1,1) has been used.
From the diagnostic it can be seen that even thoug Ljung-Box gives autocorrelaiton in standardized residuals
the correlogram shows no dangerous autocorrelation actually. Moreover via GARCH finally, unlike ARIMA, model
are able to get rid of ARCH effects in residuals, indeed the ARCH test confirm the lack of conditional
heteroskedasticity in standardized residuals.'''

# %% Unconditional volatility and forecasts

w_msft = res_msft.params[0]  # omega
a_msft = res_msft.params[1]  # alpha
b_msft = res_msft.params[3]  # beta
g_msft = res_msft.params[2]  # gamma

w_intc = res_intc.params[0]  # omega
a_intc = res_intc.params[1]  # alpha
b_intc = res_intc.params[2]  # beta

# unconditionalk volatility for a GJR(1,1)
v_msft = (w_msft / (1 - a_msft - b_msft - 0.5 * g_msft))**0.5
# unconditional volatility for a GARCH(1,1)
v_intc = (w_intc / (1 - a_intc - b_intc))**0.5
v_msft_s = r_msft.std() * 100  # historical volatility
v_intc_s = r_intc.std() * 100  # historical volatility
nobs = 247  # untill the end of 2021
date = pd.date_range(r_msft.index[-1], periods=nobs).tolist()[1:]
fc_msft = np.transpose(
    res_msft.forecast(
        horizon=nobs -
        1).variance.dropna())**0.5  # volatility forecasts
fc_msft.index = date
fc_msft.columns = ['MSFT Forecast']
fc_intc = np.transpose(
    res_intc.forecast(
        horizon=nobs -
        1).variance.dropna())**0.5  # volatility forecasts
fc_intc.index = date
fc_intc.columns = ['INTC Forecast']
v_msft = pd.DataFrame(np.ones((nobs - 1, 1)) * v_msft,
                      index=date, columns=['MSFT unconditional'])
v_intc = pd.DataFrame(np.ones((nobs - 1, 1)) * v_intc,
                      index=date, columns=['INTC unconditional'])
pd.concat(
    (fc_msft,
     fc_intc,
     v_msft,
     v_intc),
    axis=1).plot(
        grid=True,
        xlabel='Date',
        ylabel='$\\sigma_t / \\sigma$',
    title='CONDITIONAL FORECASTS AND UNCONDITIONAL VOLATILITY')
# plt.savefig(path+'GARCH_forc'+'.pdf',bbox_inches='tight')
plt.show()
plt.close()


''' --------------------------------- COMMENT --------------------------------
The uncoditional volatility for both the GARCH and the GJR can be expressed in closed form
thus can be easily computed after calibrating the models. Moreover 246 daily forecasts have
been generated for both Microsoft and Intel. As it can be seen the forecasts of the models
are monotonically coverging to the unconditional values of the volatility. Indeed this
characteristic is crucial in the GARCH models that embody the mean-reverting behaviour of
of the volatility itself. Plus it worth noting that at the end the sample volatility on the
full sample is pretty close to the unconditional volatility computed via the models
Finally it is crucial to point out that both multi-step (as in this case) forecasts nor
one-step forecasts in rolling window cannot be compared to themselves or compared to realizations
because the true variance is not observed since it is a second moment and by definition is latent.
Actually recent literature is going to use realized measures like realized variance, using high
frequency data, as proxy of the true and thus comparable variance for forecasting accuracy of models
(however it is still an open debate).
'''

'''--------------------------- MULTIVARIATE ANALYSIS ------------------------------'''
# %% Cointegration analysis

l_p_msft = np.log(Data_msft['Low'])  # low log-prices
l_p_msft.name = 'Low Microsoft'
h_p_msft = np.log(Data_msft['High'])  # high log-prices
h_p_msft.name = 'High Microsoft'
l_p_intc = np.log(Data_intc['Low'])  # low log-prices
l_p_intc.name = 'Low Intel'
h_p_intc = np.log(Data_intc['High'])  # high log-prices
h_p_intc.name = 'High Intel'
l_p = pd.concat((l_p_msft, l_p_intc), axis=1)
h_p = pd.concat((h_p_msft, h_p_intc), axis=1)
l_p.plot(title='LOW MICROSOFT AND INTEL PRICES', ylabel='$p_t$', grid=True)
# plt.savefig(path+'l_p'+'.pdf',bbox_inches='tight')
plt.show()
plt.close()
h_p.plot(title='HIGH MICROSOFT AND INTEL PRICES', ylabel='$p_t$', grid=True)
# plt.savefig(path+'h_p'+'.pdf',bbox_inches='tight')
plt.show()
plt.close()
# Granger causality test --> H0: No Granger Causality - H1: Granger Causality
gr_msft = granger(l_p, l_p.columns)
# Low Microsoft is Granger causing Low Intel but not the other way around
print(gr_msft)
# Granger causality test --> H0: No Granger Causality - H1: Granger Causality
gr_intc = granger(h_p, h_p.columns)
print(gr_intc)  # High log prices are Granger causing each other at 10% level
# the r=0 is rejected thus there is Cointegration between the series
j_co_msft = coint_johansen(l_p, 0, 1)
# the r=0 is rejected thus there is Cointegration between the series
j_co_intc = coint_johansen(h_p, 0, 1)
print(
    'Engle-Granger test p-value low prices: ' + str(
        np.around(
            sm.tsa.stattools.coint(
                l_p_msft,
                l_p_intc)[1],
            2)))  # H0: No Cointegration vs H1: Cointegration
print(
    'Engle-Granger test p-value high prices: ' + str(
        np.around(
            sm.tsa.stattools.coint(
                h_p_msft,
                h_p_intc)[1],
            2)))  # H0: No Cointegration vs H1: Cointegration

''' --------------------------------- COMMENT --------------------------------
The Granger causality between low and high log prices have been investigated showing that the low price of
Microsoft is Granger causing that of Intel but not the other way around. For the high price both are Granger
causing each other at 10% level. Regarding the cointegration, prices are both I(1) and both Engle-Granger and
Johansenn procedure have been used. Both procedures show the presence of Cointegration between low prices and
high prices. Therefore a VAR modelling makes sense.'''

# %% VAR modelling between stock log-returns

r = pd.concat((r_msft, r_intc), axis=1)
print(granger(r, r.columns))  # there is Granger causality between the prices
mdl = sm.tsa.VAR(r)  # set the VAR model
lags = np.arange(0, 16, 1)  # stream of lags
ic = pd.DataFrame(index=lags, columns=['AIC', 'BIC'])
for p in lags:
    res = mdl.fit(p)
    print('Lag order: ', p)
    print('AIC: ', res.aic)
    print('BIC: ', res.bic)
    ic.iloc[p, 0], ic.iloc[p, 1] = res.aic, res.bic

# choose lag 1 based on BIC
ic.plot(grid=True, xlabel='Lag', title='INFORMATION CRITERIA VALUES')
plt.savefig(path + 'VAR_ic' + '.pdf', bbox_inches='tight')
plt.show()
plt.close()
mdl = sm.tsa.VAR(r)  # specify the model
res = mdl.fit(1)  # fit the model with 1 lag via OLS
# Besides the constant all the other parameters are significant even at 1%
# level (constant for Microsoft is at 5% level)
print(res.summary())
fig = res.plot_acorr()
# plt.savefig(path+'VAR_ACF'+'.pdf',bbox_inches='tight')
plt.show()
plt.close()
res.irf(15).plot(orth=False)
# plt.savefig(path+'IRS'+'.pdf',bbox_inches='tight')
plt.show()
plt.close()
res.fevd(10).plot()
# plt.savefig(path+'FEVD'+'.pdf',bbox_inches='tight')
plt.show()
plt.close()


''' --------------------------------- COMMENT --------------------------------
The optimal lag length of the bivariate VAR has be choosen fitting various models and
then taking the one with the lowest BIC. Thus a VAR(1) has been chosen. The model has
been fitted via OLS and aside the constant all the parameters are statistically significant
even at 1% level. The results are good and in line with the battery of tests performed above,
that were indicating both the presence of Granger causality. Finally form the correlogram of
the residuals of the VAR equations it can be seen no autocorrelation, thus corroborating the
quality of the model.'''
