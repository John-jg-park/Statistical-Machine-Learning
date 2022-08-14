#-----Strategy1----------------------------------------------------------------------------------------
from Precode import *
import numpy
data = np.load('AllSamples.npy')

k1,i_point1,k2,i_point2 = initial_S1('0471') # please replace 0111 with your last four digit of your ID

print(k1)
print(i_point1)
print(k2)
print(i_point2)


def difference(prev_centroid,new_centroid):
    d=0
    for i in range(len(prev_centroid)):
        #Euclidean distance
        d += np.linalg.norm(prev_centroid[i]-new_centroid[i])
    return d

def assign_cluster(data,prev_centroid,k):
    cluster=[] 
    for i in range(len(data)):
        distance=[] #distance between the centroid and data point
        for j in range(k):
            distance.append(np.linalg.norm(data[i] - prev_centroid[j]))
        index=np.argmin(distance) #return index when value is min
        cluster.append(index)
    return np.asarray(cluster)

def new_centroid(data,cluster,k):
    centroid = [] 
    for i in range(k):
        array=[]
        for j in range(len(data)):
            if cluster[j]==i:
                array.append(data[j])
        centroid.append(np.mean(array,axis=0))
    return np.asarray(centroid)

def sse(data,final_centroid,cluster,k):
    sse=0
    mean=[]
    for i in range(k):
        arr=[]
        for j in range(len(data)):
            if cluster[j]==i:
                arr.append(data[j])
        mean.append(np.mean(arr,axis=0))
    #print(mean)
    for i in range(k): 
        for j in range(len(data)):
            if cluster[j]==i:
                sse += (np.linalg.norm(data[j]-mean[i]))**2
    #print(sse)        
    return sse

def k_means(data, k, i_point):
    
    diff = 10 #let's assume difference between the centroids is 10
    c_prev = i_point
    
    while diff>0.01:
        cluster = assign_cluster(data,c_prev,k) #assigns the data point to respective clusters
        #print(cluster)
        c_new = new_centroid(data,cluster,k) # to compute the new centroid point
        #print(c_new)
        diff = difference(c_prev,c_new) #to compute the difference between the centroids
        #print(diff)
        c_prev=c_new #new centroid -> centroid point
    
    print('Initial Cluster Centers')
    print(i_point)
    print('Final Cluster Centers')
    print(c_prev)
    s = sse(data,c_prev,cluster,k)
    print('SSE: ', s)
    
    return s

k_means(data, k1, i_point1)

k_means(data, k2, i_point2)

#plotting the objective function values(SSE) vs number of clusters in range from 2 to 10

sse_arr = []
k=range(2,11)
for i in k:
    s1 = k_means(data, i, initial('0471',i))
    #print(s1)
    sse_arr.append(s1)
plt.plot(k, sse_arr, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()


#-----Strategy2---------------------------------------------------------------------------------------

from Precode2 import *
import numpy
data = np.load('AllSamples.npy')

k1,i_point1,k2,i_point2 = initial_S2('0471') # please replace 0111 with your last four digit of your ID

print(k1)
print(i_point1)
print(k2)
print(i_point2)

import math

def max_centroid(kk, i_point):
    list = [i_point]

    for k in range(kk-1):
        max = 0
        new_centroid = []
    
        for d in data:
            dis = 0
        
            for l in list:
                #print(l)
                dis += math.sqrt( (d[0]-l[0])**2 + (d[1]-l[1])**2 )
                #print(l, dis)
                if np.array_equal(d,l):
                    dis = 0
                    
            if max < dis:
                max = dis
                new_centroid = d
                
        list.append(new_centroid)
        #print(list)
        #print(max, add, list)

    arr=np.array(list)
   #print(arr)
    return arr    

max_centroid(4, i_point1)

max_centroid(6, i_point2)

def difference(prev_centroid,new_centroid):
    d=0
    for i in range(len(prev_centroid)):
        #Euclidean distance
        d += np.linalg.norm(prev_centroid[i]-new_centroid[i])
    return d

def assign_cluster(data,prev_centroid,k):
    cluster=[] 
    for i in range(len(data)):
        distance=[] #distance between the centroid and data point
        for j in range(k):
            distance.append(np.linalg.norm(data[i] - prev_centroid[j]))
        index=np.argmin(distance) #return index when value is min
        cluster.append(index)
    return np.asarray(cluster)

def new_centroid(data,cluster,k):
    centroid = [] 
    for i in range(k):
        array=[]
        for j in range(len(data)):
            if cluster[j]==i:
                array.append(data[j])
        centroid.append(np.mean(array,axis=0))
    return np.asarray(centroid)

def sse(data,final_centroid,cluster,k):
    sse=0
    mean=[]
    for i in range(k):
        arr=[]
        for j in range(len(data)):
            if cluster[j]==i:
                arr.append(data[j])
        mean.append(np.mean(arr,axis=0))
    #print(mean)
    for i in range(k): 
        for j in range(len(data)):
            if cluster[j]==i:
                sse += (np.linalg.norm(data[j]-mean[i]))**2
    #print(sse)        
    return sse

def k_means(data, k, i_point):
    
    diff = 10 #let's assume difference between the centroids is 10
    c_prev = max_centroid(k, i_point)
    
    while diff>0.01:
        cluster = assign_cluster(data,c_prev,k) #assigns the data point to respective clusters
        #print(cluster)
        c_new = new_centroid(data,cluster,k) # to compute the new centroid point
        #print(c_new)
        diff = difference(c_prev,c_new) #to compute the difference between the centroids
        #print(diff)
        c_prev=c_new #new centroid -> centroid point
    
    print('Initial Cluster Centers')
    print(max_centroid(k, i_point))
    print('Final Cluster Centers')
    print(c_prev)
    s = sse(data,c_prev,cluster,k)
    print('SSE: ', s)
    
    return s

k_means(data, 4, i_point1)

k_means(data, k2, i_point2)

#plotting the objective function values(SSE) vs number of clusters in range from 2 to 10

import matplotlib.pyplot as plt

sse_arr = []
k=range(2,11)
for i in k:
    s1 = k_means(data, i, initial('0471',i))
    #print(s1)
    sse_arr.append(s1)
plt.plot(k, sse_arr, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()


#------------------------------------------------------------------------------------------
# add fuction into precode.py

def initial(id, k):
    i = int(id)%150 
    random.seed(i+500)
    init_idx = initial_point_idx(i,k,data.shape[0])
    init_s = init_point(data, init_idx)
    return init_s

# add fuction into precode2.py
def initial(id, k):
    i = int(id)%150 
    random.seed(i+800)
    init_idx = initial_point_idx2(i,k,data.shape[0])
    init_s = data[init_idx,:]
    return init_s
