import csv
import pandas as pd

#Corrispondenza timestamp gps/immagini camera
def main():
    reader_ts=pd.read_csv('./original_lables/query/timestamps.csv')
    reader_gps=pd.read_csv('./original_lables/query/ins.csv')

    df=pd.merge_asof(reader_ts.sort_values('timestamp'), reader_gps.sort_values('timestamp'), on='timestamp', direction='nearest')
    print(df)
    df.to_csv('./labels_query.csv', index=False)
    
if __name__ == "__main__":
    main()