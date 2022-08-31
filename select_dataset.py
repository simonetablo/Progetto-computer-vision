import pandas as pd
import numpy as np

labels=pd.read_csv('labels.csv')

def distance(lat1, lat2, lon1, lon2, alt1, alt2):
    r=6371000/180*np.pi
    return np.sqrt(r**2*((lat1-lat2)**2+(np.cos(lat1*np.pi/180)*(lon1-lon2))**2)+(alt1-alt2)**2)

#distanza >= 10m (più simile possibile)

min_distance=10

sum=0
p0=labels.iloc[0]
for row in labels.index:    
    p1=labels.iloc[row]
    d1=distance(p0.latitude, p1.latitude, p0.longitude, p1.longitude, p0.altitude, p1.altitude)
    if d1>=min_distance:
        p0=labels.iloc[row]
        sum+=1
print(sum)


#Verifica lunghezza totale tragitto
sum=0
for row in labels.index[1:]:    
    p0=labels.iloc[row-1]
    p1=labels.iloc[row]
    sum+=distance(p0.latitude, p1.latitude, p0.longitude, p1.longitude, p0.altitude, p1.altitude)
print(sum)
