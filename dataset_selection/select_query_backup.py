import pandas as pd
import numpy as np
import image

def main(dataset):
    
    database=pd.read_csv('./dataset_selection/selected_database_test_2.csv')
    labels=pd.read_csv('./dataset_selection/labels_'+dataset+'.csv')
    
    #def distance(lat1, lat2, lon1, lon2, alt1, alt2):
    #    r=6371000/180*np.pi
    #    return np.sqrt(r**2*((lat1-lat2)**2+(np.cos(lat1*np.pi/180)*(lon1-lon2))**2)+(alt1-alt2)**2)
    
    def distance(n1, n2, e1, e2, h1, h2):
        return np.sqrt((n1-n2)**2+(e1-e2)**2+(h1-h2)**2)
 
    pdb0=database.iloc[0]
    min_dist=1000
    start_idx=0
    
    #for row in labels.index:
    #    print(row)
    #    count=0
    #    pq=labels.iloc[row]
    #    dist=distance(pq.northing, pdb0.northing, pq.easting, pdb0.easting, pq.altitude, pdb0.altitude)
    #    if dist<min_dist:
    #        min_dist=dist
    #        start_idx=row
    #    elif count==5:
    #        break
    #    elif dist>min_dist:
    #        count+=1
    #
    tot=[]
    serie=[]
    img_index=[]
    nserie=1
    stop=0
    n=0
    sum=0
    min_distance=0
    
    selected=pd.DataFrame(columns=labels.columns)
    
    #manually set offset (start_idx) after seeing images using image.py
    start_idx+=235
    
    
    stop_idx=labels.index.stop
    print(start_idx)
    i=start_idx
    
    while i < stop_idx:
        p1=labels.iloc[i-1]
        p0=labels.iloc[i]
        d1=distance(p0.northing, p1.northing, p0.easting, p1.easting, p0.altitude, p1.altitude)
        sum+=d1
        if sum>=min_distance:
            print(i)
            p0_f=labels.iloc[[i]]
            sum+=1
            min_distance=2
            n+=1
            p0_f
            selected=selected.append(p0_f)
            sum=0
            serie.append(nserie)
            if n==5:
                tot.append(n)
                min_distance=10
                n=0
                nserie+=1
                stop+=1
            if stop==4:
                img_index.append(i)
            elif stop==5:
                selected['serie']=serie
                selected.to_csv('./dataset_selection/selected_'+dataset+'.csv', index=False)
                stop=0
                print(img_index)
                image.plot(nserie-1, img_index)
                offset=input("Offset: ")
                i+=int(offset)
                img_index=[]
        i+=1
                
    selected['serie']=serie
    selected.to_csv('./dataset_selection/selected_'+dataset+'.csv', index=False)
    
    
    
    #queryidx_start=query.index.start
    #queryidx_stop=query.index.stop
    #queryidx=pd.RangeIndex(start=queryidx_start, stop=queryidx_stop)
    #for rowd in prova:    
    #    pdb=database.iloc[rowd]
    #    min_distance=10000
    #    major_count=-1
    #    queryidx=pd.RangeIndex(start=queryidx_start, stop=queryidx_stop)
    #    print(queryidx)
    #    for rowq in queryidx:
    #        pq=query.iloc[rowq]
    #        d1=distance(pdb.latitude, pq.latitude, pdb.longitude, pq.longitude, pdb.altitude, pq.altitude)
    #        print(d1)
    #        if d1<min_distance:
    #            ps=query.iloc[[rowq]]
    #            min_distance=d1
    #            queryidx_start=rowq
    #            if major_count==-1:
    #                major_count=0
    #        elif major_count==10:
    #            print('actual', rowq)
    #            print('start', queryidx_start)
    #            queryidx_start+=1
    #            break
    #        elif d1>min_distance and major_count<10:
    #            major_count+=1
    #
    #    selected=selected.append(ps)
    #    serie.append(nserie)
    #    n+=1
    #    if n==5:
    #        tot.append(n) 
    #        min_distance=10
    #        n=0
    #        nserie+=1

    
    
    
    #Verifica lunghezza totale tragitto
    #sum=0
    #for row in labels.index[1:]:    
    #    p0=labels.iloc[row-1]
    #    p1=labels.iloc[row]
    #    sum+=distance(p0.latitude, p1.latitude, p0.longitude, p1.longitude, p0.altitude, p1.altitude)
    #print(sum)
    
if __name__ == "__main__":
    main(dataset='database_snow')