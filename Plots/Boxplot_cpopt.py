import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
# inst = "1515"
# alpha= "0.75"
# arr_time="30"
# batch = "b_2"

# Defiene extraction function
def extract(data,alpha):
    # Creating empty lists to store the extracted data
    simulations = []
    hours = []
    ms_values = []
    de_values = []
    added_values = []
    compute_values = []

    # Looping through each simulation and extracting the required metrics
    for simulation in data:
        # print("simulation is:", data[simulation])
        for hour in data[simulation]["Ms"]:
            ms = data[simulation]["Ms"][hour]
            de = data[simulation]["DE"][hour]
            added = data[simulation]["added"][hour]
            compute =data[simulation]["compute_time"][hour]
            
            # Appending the extracted data to the respective lists
            simulations.append(simulation)
            hours.append(hour)
            ms_values.append(ms)
            de_values.append(de)
            added_values.append(added)
            compute_values.append(compute)

    # Creating a pandas DataFrame from the extracted data
    df = pd.DataFrame({
        "Simulation": simulations,
        "Hour": hours,
        "ms": ms_values,
        "de": de_values,
        "added": added_values,
        "compute_time": compute_values

    })
    fil_initial=df[df["Hour"] == 0]
    initial_ms = fil_initial["ms"].mean()
    filtered_df = df[df["added"] != 0]
    mean_ms = filtered_df["ms"].mean()
    mean_de = filtered_df["de"].mean()
    mean_time= filtered_df["compute_time"].mean()
    obj_ppo=alpha* mean_de + (1-alpha)*mean_ms

    return filtered_df,initial_ms,mean_ms, mean_de,mean_time,obj_ppo


def main(
         inst:str="1010",
         batch1:str="b_1",
         batch2:str="b_2",
         batch3:str="b_3",
         arr_time:str="30",
         ):
    location1_early="./"+inst+"/LG/"+batch1+"/30/"
    location1_intermediate="./"+inst+"/LG/"+batch1+"/60/"
    location1_late="./"+inst+"/LG/"+batch1+"/90/"


    location2_early="./"+inst+"/LG/"+batch2+"/30/"
    location2_intermediate="./"+inst+"/LG/"+batch2+"/60/"
    location2_late="./"+inst+"/LG/"+batch2+"/90/"

    location3_early="./"+inst+"/LG/"+batch3+"/30/"
    location3_intermediate="./"+inst+"/LG/"+batch3+"/60/"
    location3_late="./"+inst+"/LG/"+batch3+"/90/"

   
    
    # Assuming your PPO dictionary of simulations is named "data"
    data1_early = open(location1_early+"cplex_data_50.pkl", "rb")
    data1_early= data1_early.read()
    data1_early = pickle.loads(data1_early)

    data1_intermediate = open(location1_intermediate+"cplex_data_50.pkl", "rb")
    data1_intermediate= data1_intermediate.read()
    data1_intermediate= pickle.loads(data1_intermediate)

    data1_late = open(location1_late+"cplex_data_50.pkl", "rb")
    data1_late= data1_late.read()
    data1_late = pickle.loads(data1_late)


    data2_early = open(location2_early+"cplex_data_50.pkl", "rb")
    data2_early= data2_early.read()
    data2_early = pickle.loads(data2_early)

    data2_intermediate = open(location2_intermediate+"cplex_data_50.pkl", "rb")
    data2_intermediate= data2_intermediate.read()
    data2_intermediate= pickle.loads(data2_intermediate)

    data2_late = open(location2_late+"cplex_data_50.pkl", "rb")
    data2_late= data2_late.read()
    data2_late = pickle.loads(data2_late)


    data3_early = open(location3_early+"cplex_data_50.pkl", "rb")
    data3_early= data3_early.read()
    data3_early = pickle.loads(data3_early)

    data3_intermediate = open(location3_intermediate+"cplex_data_50.pkl", "rb")
    data3_intermediate= data3_intermediate.read()
    data3_intermediate= pickle.loads(data3_intermediate)

    data3_late = open(location3_late+"cplex_data_50.pkl", "rb")
    data3_late= data3_late.read()
    data3_late = pickle.loads(data3_late)



    df1_early, _, _, _, _, _ = extract(data1_early, 1)
    df1_intermediate, initial_ms1, mean_ms, mean_de, mean_time, obj_ppo = extract(data1_intermediate, 1)
    df1_late, initial_ms, mean_ms, mean_de, mean_time, obj_ppo = extract(data1_late, 1)
    dfs1 = [df1_early, df1_intermediate, df1_late]

    df2_early, _, _, _, _, _ = extract(data2_early, 1)
    df2_intermediate, initial_ms2, mean_ms, mean_de, mean_time, obj_ppo = extract(data2_intermediate, 1)
    df2_late, initial_ms, mean_ms, mean_de, mean_time, obj_ppo = extract(data2_late, 1)
    dfs2 = [df2_early, df2_intermediate, df2_late]

    df3_early, _, _, _, _, _ = extract(data3_early, 1)
    df3_intermediate, initial_ms3, mean_ms, mean_de, mean_time, obj_ppo = extract(data3_intermediate, 1)
    df3_late, initial_ms, mean_ms, mean_de, mean_time, obj_ppo = extract(data3_late, 1)
    dfs3 = [df3_early, df3_intermediate, df3_late]


    # Caracteristics of boxplots
    labels = ['Early', 'Intermediate', 'Late']
    sns.set_style("whitegrid")  # "white","dark","darkgrid","ticks"
    # sns.set_style("ticks")
    palette = ['#FF2709', '#09FF10', '#0030D7', '#FA70B5']


    boxprops = dict(linestyle='-', linewidth=1.5, color='#00145A')
    flierprops = dict(marker='o', markersize=1,
                    linestyle='none')
    whiskerprops = dict(color='#00145A')
    capprops = dict(color='#00145A')
    medianprops = dict(linewidth=1.5, linestyle='-', color='#01FBEE')

   # Create the box plots
    plt.figure(figsize=(12, 6))
    # Box plot for Makespan (ms), Batch=1
    plt.subplot(1, 3, 1)
    plt.boxplot([df['ms'] for df in dfs1], labels=labels,notch=False, boxprops=boxprops, whiskerprops=whiskerprops,capprops=capprops, flierprops=flierprops, medianprops=medianprops,showmeans=True)
    plt.title('MSP per arrival time, Batch=1')
    plt.xlabel('Arrival time')
    plt.ylabel('Makespan (MSP)')
    plt.axhline(y=initial_ms, color='#ff3300', linestyle='--', linewidth=1, label='MS_initial')
    plt.legend()
    # Box plot for Makespan (ms), Batch=2
    
    plt.subplot(1, 3, 2)
    plt.boxplot([df['ms'] for df in dfs2], labels=labels,notch=False, boxprops=boxprops, whiskerprops=whiskerprops,capprops=capprops, flierprops=flierprops, medianprops=medianprops,showmeans=True)
    plt.title('MSP per arrival time, Batch=2')
    plt.xlabel('Arrival time')
    plt.ylabel('Makespan (MSP)')
    plt.axhline(y=initial_ms, color='#ff3300', linestyle='--', linewidth=1, label='MS_initial')
    plt.legend()

    # Box plot for Makespan (ms), Batch=3
    plt.subplot(1, 3, 3)
    plt.boxplot([df['ms'] for df in dfs3], labels=labels,notch=False, boxprops=boxprops, whiskerprops=whiskerprops,capprops=capprops, flierprops=flierprops, medianprops=medianprops,showmeans=True)
    plt.title('MSP per arrival time, Batch=3')
    plt.xlabel('Arrival time')
    plt.ylabel('Makespan (MSP)')
    plt.axhline(y=initial_ms, color='#ff3300', linestyle='--', linewidth=1, label='MS_initial')
    plt.legend()


    plt.suptitle('Makespan distribution per arrival time accross batch sizes (generated by CPOPTIMIZER)')
    
    plt.tight_layout()
    
    plt.show()


    #  ******************************************************DE boxplot
    # Box plot for Total Deviation (de)
    # plt.subplot(1, 3, 1)
    # plt.boxplot([df['de'] for df in dfs1], labels=labels,notch=False, boxprops=boxprops, whiskerprops=whiskerprops,capprops=capprops, flierprops=flierprops, medianprops=medianprops,showmeans=True)
    # plt.title('DEV distribution per arrival time, Batch=1')
    # plt.xlabel('Arrival time')
    # plt.ylabel('Total Deviation (DEV)')

    # plt.subplot(1, 3, 2)
    # plt.boxplot([df['de'] for df in dfs2], labels=labels,notch=False, boxprops=boxprops, whiskerprops=whiskerprops,capprops=capprops, flierprops=flierprops, medianprops=medianprops,showmeans=True)
    # plt.title('DEV distribution per arrival time,, Batch=2')
    # plt.xlabel('Arrival time')
    # plt.ylabel('Total Deviation (DEV)')


    # plt.subplot(1, 3, 3)
    # plt.boxplot([df['de'] for df in dfs3], labels=labels,notch=False, boxprops=boxprops, whiskerprops=whiskerprops,capprops=capprops, flierprops=flierprops, medianprops=medianprops,showmeans=True)
    # plt.title('DEV distribution per arrival time, Batch=3')
    # plt.xlabel('Arrival time')
    # plt.ylabel('Total Deviation (DEV)')

    
    # plt.suptitle('Total deviation distribution per arrival time accross batch sizes (generated by CPOPTIMIZER)')
    # plt.tight_layout()
    # plt.show()

   
inst="1010"
print("Boxplot:", inst)
main(inst=inst, batch1="b_1",batch2="b_2",batch3="b_3", arr_time="30")

    # initial_ms,mean_ms_cp_1, mean_de_cp_1,mean_time_cp_1,obj_cp_1=extract(cplex_data_1,1)
 
    # initial_ms,mean_ms_cp_0, mean_de_cp_0,mean_time_cp_0,obj_cp_0=extract(cplex_data_0,0.0)

    # initial_ms,mean_ms_cp_50, mean_de_cp_50,mean_time_cp_50,obj_cp_50=extract(cplex_data_50,0.5)
  
    # initial_ms,mean_ms_spt, mean_de_spt,mean_time_spt,obj_spt=extract(SPT_data,1.0)   