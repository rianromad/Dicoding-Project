#mengimpor library
import os, zipfile
import kaggle
import pandas as pd
from datetime import timedelta, date
import numpy as np

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose as sd

from prophet import Prophet
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.tsa.api as smt

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from prophet.diagnostics import cross_validation, performance_metrics


import warnings
warnings.filterwarnings("ignore")

#akses dataset melalui kaggle
#!kaggle datasets download -d divyajeetthakur/walmart-sales-prediction

#ekstrak file
#path = 'walmart-sales-prediction.zip'
#zip_ref = zipfile.ZipFile(path, 'r')
#zip_ref.extractall()
#zip_ref.close()

#membuka masing-masing file
train = pd.read_csv('train.csv')
stores = pd.read_csv('stores.csv')
features = pd.read_csv('features.csv')

#melihat data train
train.head(3)

#melihat data stores
stores.head(3)

#melihat data features
features.head(3)

#menggabungkan data train dengan stores dan features
df_train = train.merge(stores, how='left').merge(features, how='left')
df_train.head(3)

#mengetahui tipe data masing-masing atribut
df_train.info()

df_train['Date'] = pd.to_datetime(df_train['Date'])

#cek kembali tipe data Date
df_train.info()

#mengecek missing value
df_train.isnull().sum()

#statistik deskriptif
df_train.describe().T

#sebaran penjualan mingguan
fig, ax = plt.subplots(figsize=(20,5))
sns.histplot(df_train['Weekly_Sales'], bins=30, kde=True, ax=ax)
ax.set_title('Distribusi Weekly_Sales')
plt.show()

#mengganti missing value dengan nilai 0
df_train = df_train.fillna(0)

#mengecek kembali missing value
df_train.isnull().sum()

#Transformasi atribut IsHoliday
df_train['IsHoliday'] = df_train['IsHoliday'].replace({True:1, False:0}) 
df_train = pd.get_dummies(df_train)

df_train.info()

#Membuat variabel ts yang berisi tanggal dan weekly_sales yang sudah dilakukan grouping
ts = df_train.copy().groupby('Date').agg('sum')['Weekly_Sales']
ts.head(3)

#plot time series
fig, ax = plt.subplots(figsize=(20,6))
ax.plot(ts.index,ts.values)
ax.set_title("Plot Total Penjualan Mingguan")
ax.set_xlabel("Tanggal")
ax.set_ylabel("Total Penjualan Mingguan")
ax.set_yscale("log")
plt.show()

#plot dekomposisi
result = sm.tsa.seasonal_decompose(ts.values,period=52,model="multiplicative")
result.plot()
plt.show()

### FB PROPHET ###
#ubah nama kolom
prophet_ts = ts.reset_index()
prophet_ts.columns = ["ds","y"]
prophet_ts.head(5)

# fit model prophet
prophet_m = Prophet(yearly_seasonality = True)
prophet_m.fit(prophet_ts)

#selisih hari dari tanggal 6 Januari 2012 sampai 26 Oktober 2012 
print(date(2012,10,26)-date(2012,1,6))

#evaluasi model dilakukan dengan data dari tanggal 6 Januari 2012
cutoffs = pd.to_datetime(['2012-01-6'])
df_cv = cross_validation(prophet_m, cutoffs=cutoffs, horizon='294 days')

#evaluasi model FB Prophet dari tanggal 6 Januari 2012 dst
mse_p = float(performance_metrics(df_cv).tail(1)['mse'])
mae_p = float(performance_metrics(df_cv).tail(1)['mae'])
mape_p = float(performance_metrics(df_cv).tail(1)['mape'])

print(float(mape_p))

#prediksi 26 minggu ke depan
future = prophet_m.make_future_dataframe(periods=26, freq='W')
prophet_forecast = prophet_m.predict(future)
prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#plot hasil prediksi fb prophet -> untuk dibandingkan dengan model sarima
fig, ax = plt.subplots(figsize=(20,6))
ax.plot(ts, label="Actual")
ax.plot(prophet_forecast.set_index('ds')['yhat']['2011-02-11':], label="Forecast")
ax.set_title("Walmart Sales Prediction (FB Prophet Model)")
ax.set_xlabel("Date")
ax.set_ylabel("Total Weekly Sales")
ax.set_yscale("log")
fig.legend()
plt.show()

### SARIMA ###

#fungsi untuk plot acf pacf

def ts_plot(ts, lags):
    with plt.style.context("bmh"):    
        fig = plt.figure(figsize=(12, 7))
        ts_ax = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        acf_ax = plt.subplot2grid((2, 2), (1, 0))
        pacf_ax = plt.subplot2grid((2, 2), (1, 1))
        ts.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(ts)[1]
        ts_ax.set_title('Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(ts, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(ts, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        
ts_plot(ts, 26)

#seasonal differencing -> D = 1
ts_diff = ts - ts.shift(52)
ts_plot(ts_diff[52:], 26)

#differencing -> d = 1
ts_diff = ts_diff - ts_diff.shift(1)
ts_plot(ts_diff[52+1:], 26)

# fit model SARIMA
p, d, q = 2, 1, 3
P, D, Q = 2, 1, 3
s = 52
sarima_m =sm.tsa.statespace.SARIMAX(ts, order=(p, d, q), seasonal_order=(P, D, Q, s)).fit(disp=-1)
print(sarima_m.summary())

#plot residu
ts_plot(sarima_m.resid[24+1:], lags=40)

sarima = pd.DataFrame(ts)
sarima.columns = ['actual']
sarima['model'] = sarima_m.fittedvalues
sarima['model'][:s+d] = np.NaN

#prediksi 26 minggu ke depan
sarima_forecast = sarima_m.predict(start = sarima.shape[0], end = sarima.shape[0]+26)
sarima_forecast = sarima.model.append(sarima_forecast)

#ambil data dari tanggal 6 Januari 2012 sampai 26 Oktober 2012 untuk evaluasi
sarima_ev = sarima['2012-01-06':]
mse_sarima = mean_squared_error(sarima_ev['actual'],sarima_ev['model'])
mae_sarima = mean_absolute_error(sarima_ev['actual'],sarima_ev['model'])
mape_sarima = mean_absolute_percentage_error(sarima_ev['actual'],sarima_ev['model'])

print(mape_sarima)

#plot hasil prediksi sarima

fig, ax = plt.subplots(figsize=(20,6))
ax.plot(sarima['actual'], label="Actual")
ax.plot(sarima_forecast, label="Forecast")
ax.set_title("Walmart Sales Prediction (SARIMA Model)")
ax.set_xlabel("Date")
ax.set_ylabel("Total Weekly Sales")
ax.set_yscale("log")
fig.legend()
plt.show()


### EVALUASI ###
#tabel perbandingan performa model

score = pd.DataFrame({"Model":["FB Prophet","SARIMA"],
                      "MAE Score":[mae_p, mae_sarima],
                      "MSE Score":[mse_p, mse_sarima],
                      "MAPE Score":[mape_p,mape_sarima]}).set_index("Model")
print(score)

fig, ax = plt.subplots(figsize=(20,6))
ax.plot(sarima_forecast['2012-10-26':])
ax.set_title("Prediksi Total Penjualan untuk 26 Minggu ke depan")
ax.set_xlabel("Date")
ax.set_ylabel("Total Weekly Sales")
ax.set_yscale("log")
plt.show()