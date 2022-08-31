import csv
import pandas as pd

#Corrispondenza timestamp gps/immagini camera

reader_ts=pd.read_csv('./original_lables/timestamps.csv')
reader_gps=pd.read_csv('./original_lables/ins.csv')

df=pd.merge_asof(reader_ts.sort_values('timestamp'), reader_gps.sort_values('timestamp'), on='timestamp', direction='nearest')
print(df)
df.to_csv('labels.csv', index=False)