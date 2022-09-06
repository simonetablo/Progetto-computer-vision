import pandas as pd
import numpy as np
def main(dataset):
    labels=pd.read_csv('./dataset_selection/labels_'+dataset+'.csv')
    
    #def distance(lat1, lat2, lon1, lon2, alt1, alt2):
    #    r=6371000/180*np.pi
    #    return np.sqrt(r**2*((lat1-lat2)**2+(np.cos(lat1*np.pi/180)*(lon1-lon2))**2)+(alt1-alt2)**2)
    
    def distance(n1, n2, e1, e2, h1, h2):
        return np.sqrt((n1-n2)**2+(e1-e2)**2+(h1-h2)**2)
    
    #distanza >= 10m (più simile possibile)
    
    min_distance=10
    
    n=0
    sum=0
    tot=[]
    #p0=labels.iloc[0]
    serie=[]
    nserie=1
    
    selected=pd.DataFrame(columns=labels.columns)

    range=pd.RangeIndex(start=1, stop=labels.index.stop)
    
    for row in range:    
        p1=labels.iloc[row-1]
        p0=labels.iloc[row]
        #d1=distance(p0.latitude, p1.latitude, p0.longitude, p1.longitude, p0.altitude, p1.altitude)
        
        d1=distance(p0.northing, p1.northing, p0.easting, p1.easting, p0.altitude, p1.altitude)
        sum+=d1
        if sum>=min_distance:
            p0_f=labels.iloc[[row]]
            sum+=1
            min_distance=2
            n+=1
            p0_f
            selected=selected.append(p0_f)
            serie.append(nserie)
            sum=0
            if n==5:
                tot.append(n) 
                min_distance=10
                n=0
                nserie+=1
                
    selected['serie']=serie
    selected.to_csv('./dataset_selection/selected_'+dataset+'.csv', index=False)
    
    
    
    #Verifica lunghezza totale tragitto
    #sum=0
    #for row in labels.index[1:]:    
    #    p0=labels.iloc[row-1]
    #    p1=labels.iloc[row]
    #    sum+=distance(p0.latitude, p1.latitude, p0.longitude, p1.longitude, p0.altitude, p1.altitude)
    #print(sum)
    
if __name__ == "__main__":
    main(dataset='night')