# This script reads the Scats-Data-October-2006.xls file and processes the data to create a csv file that will help
# training the AI Agent to predict the traffic flow.

import pandas as pd

dataframe = pd.read_excel('Scats-Data-October-2006.xls',sheet_name='Data',header=1)

dataframe['SCATS Number'] = dataframe['SCATS Number'].astype(str)

filtered_df = dataframe[dataframe['SCATS Number'] == '2200']

time_intervals = [f'V{i:02}' for i in range(96)]    # read V00 - V95 in data sheet

selected_columns = filtered_df[['SCATS Number','Date','Location'] + time_intervals]

df_melted = selected_columns.melt(
    id_vars=['SCATS Number','Date','Location'],
    var_name='Interval',
    value_name='Lane Flow (Veh/15 Minutes)'
)

df_melted['Interval'] = df_melted['Interval'].str[1:].astype(int)

df_melted['Time'] = pd.to_datetime(df_melted['Date']) + pd.to_timedelta(df_melted['Interval'] *15, unit='min')


df_melted['direction'] = df_melted['Location'].str.extract(r'(\b[E|W|N|S]\b)')

df_melted = df_melted[['Time','Lane Flow (Veh/15 Minutes)','direction']]
df_melted = df_melted.sort_values(by=['direction','Time']).reset_index(drop=True)

print(df_melted)
df_melted.to_csv("data/2200_flow_processed.csv", index=False)
