import DF
import numpy as np
import time
import pandas as pd
from sklearn.metrics import mean_squared_error

data = pd.read_csv('../data/GPS.csv',index_col=None,header=0)
print("??")
data.iloc[:,0]=data.iloc[:,0]-data.iloc[0,0]
X = np.array(data["time"])
obs = np.array(data[["sensor1","sensor2","sensor3","sensor4"]])
true = np.array(data["true"])

n = obs.shape[0]
m = obs.shape[1]
print("n = ",n,"  m = ",m)

print("===== start DFDP =====")
start_time = time.time() 
sel1,_ = DF.DFDP(obs,X,5)
end_time = time.time()
print("[DFDP] time cost:",end_time-start_time,'loss:',mean_squared_error(true,sel1))

print("===== start DFRC =====")
start_time2 = time.time() 
sel2,_,_ = DF.DFRC(obs,X,3)
end_time2 = time.time()
print("[DFRC] time cost:",end_time2-start_time2,'loss:',mean_squared_error(true,sel2))

print("===== start DFRT =====")
start_time3 = time.time() 
sel3 = DF.DFRT(obs,X,4)
end_time3 = time.time()
print("[DFRT] time cost:",end_time3-start_time3,'loss:',mean_squared_error(true,sel3))

