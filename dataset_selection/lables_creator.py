import csv
import pandas as pd

#Corrispondenza timestamp gps/immagini camera
def main(dataset):
    reader_ts=pd.read_csv('./original_lables/'+dataset+'/timestamps.csv')
    reader_gps=pd.read_csv('./original_lables/'+dataset+'/ins.csv')

    df=pd.merge_asof(reader_ts.sort_values('timestamp'), reader_gps.sort_values('timestamp'), on='timestamp', direction='nearest')
    print(df)
    df.to_csv('./labels_'+dataset+'.csv', index=False)
    
if __name__ == "__main__":
    main(dataset='query_test_2')