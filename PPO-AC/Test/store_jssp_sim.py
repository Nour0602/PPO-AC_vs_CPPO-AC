from aggr_mat import *
from utils import *
import numpy as np  
from reset_set_env import SJSSP 
from Inst_generator import uni_instance_gen
import pickle
np.set_printoptions(suppress=True)
from generate_ref_cp import*


N_JOBS_N =3
print(  N_JOBS_N)
N_MACHINES_N = 3
print(N_MACHINES_N)
data_generator = uni_instance_gen

list_store_sim={}
N_sim=25
for simulation in range(N_sim):
     
     dur,seq=data_generator(n_j=N_JOBS_N, n_m=N_MACHINES_N, low=1, high=99)
     data1,data2,start,mchIds,end, sol=optimize_and_visualize(N_MACHINES_N,N_JOBS_N,dur,seq, 2, 1)
     data=(data1,data2,mchIds,start,end,sol)
     list_store_sim[simulation]=data
inst = "10050"
arrival = "90"
alpha= "0.75"
batch = "b_1"
# location="./"+inst
location="./"
print("location is:", location)
list_store_file=open(location+"/store_data.pkl", "wb")
pickle.dump(list_store_sim, list_store_file)
list_store_file.close() 