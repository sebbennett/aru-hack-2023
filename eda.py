import pandas as pd

ISRAEL_PUBLIC_HOLIDAYS = pd.read_html('./israel_holidays.html')[0]

ISRAEL_PUBLIC_HOLIDAYS['full_date'] = ISRAEL_PUBLIC_HOLIDAYS['Date'] + ' 2020'

ISRAEL_PUBLIC_HOLIDAYS['full_date_datetime'] = pd.to_datetime(ISRAEL_PUBLIC_HOLIDAYS['full_date'], format='%b %d %Y')




df = pd.read_csv("./data/corona_tested_individuals_ver_0083.english.csv.zip")


df['covid_mapped'] = df['corona_result'].map({
    'negative':0,
    'positive':1,
    'other':-1
})

df['test_indication_mapped'] = df['test_indication'].map({
    'Other':-1,
    'Contact with confirmed':1,
    'Abroad':0
})

df['test_date'] = pd.to_datetime(df['test_date'])
df['julian_date'] = df.test_date.dt.dayofyear#pd.DatetimeIndex(df.test_date).to_julian_date()

df_corr = df.corr()

import seaborn as sns
sns.heatmap(df_corr, annot=True)


import matplotlib.pyplot as plt
plot_df = df.groupby(['test_date','corona_result']).size().reset_index().pivot(index='test_date',columns='corona_result')
plot_df.columns = ['negative','other','positive']



ax = plot_df['positive'].plot()
for value in ISRAEL_PUBLIC_HOLIDAYS['full_date_datetime'].unique():
    ax.axvline(value, color='red', linestyle='--')





sns.scatterplot(df, x='test_date',y='julian_date')

for column in ['cough', 'fever', 'sore_throat', 'shortness_of_breath', 'head_ache','age_60_and_above', 'gender', 'test_indication']:
    print(f'column = {column}')
    print(df.groupby(column)['corona_result'].value_counts(normalize=True))
    print('='*50)
    
    
from statsmodels.tsa.seasonal import seasonal_decompose

result= seasonal_decompose(plot_df['positive'], model='multiplicable', period=7)
result.plot()


ax = result.trend.plot()
for value in ISRAEL_PUBLIC_HOLIDAYS['full_date_datetime'].unique():
    ax.axvline(value, color='red', linestyle='--')
    
    
    
    
plot_df['positive_rate'] = plot_df['positive'] / plot_df[['negative','positive','other']].sum(axis=1)
ax = plot_df['positive_rate'].plot()
for value in ISRAEL_PUBLIC_HOLIDAYS['full_date_datetime'].unique():
    ax.axvline(value, color='red', linestyle='--')
    
result= seasonal_decompose(plot_df['positive_rate'], model='multiplicable', period=7)
result.plot()

ax = result.trend.plot()
for value in ISRAEL_PUBLIC_HOLIDAYS['full_date_datetime'].unique():
    ax.axvline(value, color='red', linestyle='--')