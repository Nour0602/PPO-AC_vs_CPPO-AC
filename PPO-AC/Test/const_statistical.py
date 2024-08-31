import pandas as pd
import pickle
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

    return initial_ms,mean_ms, mean_de,mean_time,obj_ppo


def main(
         inst:str="1010",
         batch:str="b_2",
         arr_time:str="30",
         ):
    location="./"+inst+"/LG/"+batch+"/"+arr_time+"/"
    # lc_spt="./1515"+"/LG/"+batch+"/"+arr_time+"/"
    lc_spt1="./1515"+"/NOLG/"+batch+"/"+arr_time+"/"

    print("location is:", location)
    # import matplotlib.pyplot as plt
    # Assuming your PPO dictionary of simulations is named "data"
    data = open(location+"dico_data_multi.pkl", "rb")
    data = data.read()
    data = pickle.loads(data)

    # Assuming your CPLEX dictionary of simulations is named "cplex_data"
    # cplex_data = open("cplex_data.pkl", "rb")
    cplex_data_1 = open(location+"cplex_data_1.pkl", "rb")
    cplex_data_1 = cplex_data_1.read()
    cplex_data_1 = pickle.loads(cplex_data_1)
 
    cplex_data_0 = open(location+"cplex_data_0.pkl", "rb")
    cplex_data_0 = cplex_data_0.read()
    cplex_data_0 = pickle.loads(cplex_data_0)

    cplex_data_50 = open(location+"cplex_data_50.pkl", "rb")
    cplex_data_50 = cplex_data_50.read()
    cplex_data_50 = pickle.loads(cplex_data_50)

    SPT_data = open(lc_spt1+"SPT_data.pkl", "rb")
    SPT_data = SPT_data.read()
    SPT_data = pickle.loads(SPT_data)
    
    initial_ms,mean_ms, mean_de,mean_time,obj_ppo=extract(data,1)

    initial_ms1,mean_ms_cp_1, mean_de_cp_1,mean_time_cp_1,obj_cp_1=extract(cplex_data_1,1)
 
    initial_ms11,mean_ms_cp_0, mean_de_cp_0,mean_time_cp_0,obj_cp_0=extract(cplex_data_0,0.0)

    initial_ms1,mean_ms_cp_50, mean_de_cp_50,mean_time_cp_50,obj_cp_50=extract(cplex_data_50,0.75)
  
    initial_ms1,mean_ms_spt, mean_de_spt,mean_time_spt,obj_spt=extract(SPT_data,1.0)


    print("initial ms is:", initial_ms)
    if mean_de_cp_1!=0 and mean_de_cp_0!=0 and mean_de_cp_50!=0:
        comparison_df = pd.DataFrame({
            "Method": ["PPO", "CPOPT_1" ,"CPOPT_0","CPOPT_50","SPT","Gap_cp_1(%)", "Gap_cp_0(%)", "Gap_cp_50(%)", "Gap_SPT(%)" ],
            "Mean_ms": [mean_ms, mean_ms_cp_1,mean_ms_cp_0,mean_ms_cp_50,mean_ms_spt,((mean_ms-mean_ms_cp_1) / mean_ms_cp_1)*100,((mean_ms-mean_ms_cp_0) / mean_ms_cp_0)*100,((mean_ms-mean_ms_cp_50) / mean_ms_cp_50)*100,( (mean_ms-mean_ms_spt) / mean_ms_spt)*100],
            "Mean_deviation": [mean_de, mean_de_cp_1,mean_de_cp_0,mean_de_cp_50,mean_de_spt, ((mean_de-mean_de_cp_1) / mean_de_cp_1)*100,((mean_de-mean_de_cp_0) / mean_de_cp_0)*100,((mean_de-mean_de_cp_50) / mean_de_cp_50)*100,((mean_de-obj_spt) / obj_spt)*100],
            "Mean_computational_time":[mean_time,mean_time_cp_1,mean_time_cp_0,mean_time_cp_50,mean_time_spt,"-","-","-","-"],
            "Objective_function":[obj_ppo ,obj_cp_1 ,obj_cp_0,obj_cp_50,obj_spt,"-","-","-","-"],
            "Initial_MS":[initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,"-","-"]
            
        })
    elif mean_de_cp_1==0 and mean_de_cp_0==0 and mean_de_cp_50==0:
        comparison_df = pd.DataFrame({
            "Method": ["PPO", "CPOPT_1" ,"CPOPT_0","CPOPT_50","SPT","Gap_cp_1(%)", "Gap_cp_0(%)", "Gap_cp_50(%)", "Gap_SPT(%)" ],
            "Mean_ms": [mean_ms, mean_ms_cp_1,mean_ms_cp_0,mean_ms_cp_50,mean_ms_spt,((mean_ms-mean_ms_cp_1) / mean_ms_cp_1)*100,((mean_ms-mean_ms_cp_0) / mean_ms_cp_0)*100,((mean_ms-mean_ms_cp_50) / mean_ms_cp_50)*100,( (mean_ms-mean_ms_spt) / mean_ms_spt)*100],
            "Mean_deviation": [mean_de, mean_de_cp_1,mean_de_cp_0,mean_de_cp_50,mean_de_spt, "-","-","-","-"],
            "Mean_computational_time":[mean_time,mean_time_cp_1,mean_time_cp_0,mean_time_cp_50,mean_time_spt,"-","-","-","-"],
            "Objective_function":[obj_ppo ,obj_cp_1 ,obj_cp_0,obj_cp_50,obj_spt,"-","-","-","-"],
            "Initial_MS":[initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,"-","-"]
            
        })
    elif mean_de_cp_1==0 and mean_de_cp_0!=0 and mean_de_cp_50!=0:
        comparison_df = pd.DataFrame({
            "Method": ["PPO", "CPOPT_1" ,"CPOPT_0","CPOPT_50","SPT","Gap_cp_1(%)", "Gap_cp_0(%)", "Gap_cp_50(%)", "Gap_SPT(%)" ],
            "Mean_ms": [mean_ms, mean_ms_cp_1,mean_ms_cp_0,mean_ms_cp_50,mean_ms_spt,((mean_ms-mean_ms_cp_1) / mean_ms_cp_1)*100,((mean_ms-mean_ms_cp_0) / mean_ms_cp_0)*100,((mean_ms-mean_ms_cp_50) / mean_ms_cp_50)*100,( (mean_ms-mean_ms_spt) / mean_ms_spt)*100],
            "Mean_deviation": [mean_de, mean_de_cp_1,mean_de_cp_0,mean_de_cp_50,mean_de_spt, "-",((mean_de-mean_de_cp_0) / mean_de_cp_0)*100,((mean_de-mean_de_cp_50) / mean_de_cp_50)*100,((mean_de-obj_spt) / obj_spt)*100],
            "Mean_computational_time":[mean_time,mean_time_cp_1,mean_time_cp_0,mean_time_cp_50,mean_time_spt,"-","-","-","-"],
            "Objective_function":[obj_ppo ,obj_cp_1 ,obj_cp_0,obj_cp_50,obj_spt,"-","-","-","-"],
            "Initial_MS":[initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,"-","-"]
            
        })
    elif mean_de_cp_1==0 and mean_de_cp_0==0 and mean_de_cp_50!=0:
        comparison_df = pd.DataFrame({
            "Method": ["PPO", "CPOPT_1" ,"CPOPT_0","CPOPT_50","SPT","Gap_cp_1(%)", "Gap_cp_0(%)", "Gap_cp_50(%)", "Gap_SPT(%)" ],
            "Mean_ms": [mean_ms, mean_ms_cp_1,mean_ms_cp_0,mean_ms_cp_50,mean_ms_spt,((mean_ms-mean_ms_cp_1) / mean_ms_cp_1)*100,((mean_ms-mean_ms_cp_0) / mean_ms_cp_0)*100,((mean_ms-mean_ms_cp_50) / mean_ms_cp_50)*100,( (mean_ms-mean_ms_spt) / mean_ms_spt)*100],
            "Mean_deviation": [mean_de, mean_de_cp_1,mean_de_cp_0,mean_de_cp_50,mean_de_spt, "-","-",((mean_de-mean_de_cp_50) / mean_de_cp_50)*100,((mean_de-obj_spt) / obj_spt)*100],
            "Mean_computational_time":[mean_time,mean_time_cp_1,mean_time_cp_0,mean_time_cp_50,mean_time_spt,"-","-","-","-"],
            "Objective_function":[obj_ppo ,obj_cp_1 ,obj_cp_0,obj_cp_50,obj_spt,"-","-","-","-"],
            "Initial_MS":[initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,"-","-"]
            
        })
    elif mean_de_cp_1==0 and mean_de_cp_0!=0 and mean_de_cp_50==0:
        comparison_df = pd.DataFrame({
            "Method": ["PPO", "CPOPT_1" ,"CPOPT_0","CPOPT_50","SPT","Gap_cp_1(%)", "Gap_cp_0(%)", "Gap_cp_50(%)", "Gap_SPT(%)" ],
            "Mean_ms": [mean_ms, mean_ms_cp_1,mean_ms_cp_0,mean_ms_cp_50,mean_ms_spt,((mean_ms-mean_ms_cp_1) / mean_ms_cp_1)*100,((mean_ms-mean_ms_cp_0) / mean_ms_cp_0)*100,((mean_ms-mean_ms_cp_50) / mean_ms_cp_50)*100,( (mean_ms-mean_ms_spt) / mean_ms_spt)*100],
            "Mean_deviation": [mean_de, mean_de_cp_1,mean_de_cp_0,mean_de_cp_50,mean_de_spt, "-",((obj_ppo-mean_de_cp_0) / mean_de_cp_0)*100,"-",((obj_ppo-obj_spt) / obj_spt)*100],
            "Mean_computational_time":[mean_time,mean_time_cp_1,mean_time_cp_0,mean_time_cp_50,mean_time_spt,"-","-","-","-"],
            "Objective_function":[obj_ppo ,obj_cp_1 ,obj_cp_0,obj_cp_50,obj_spt,"-","-","-","-"],
            "Initial_MS":[initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,"-","-"]
            
        })
    elif mean_de_cp_1!=0 and mean_de_cp_0==0 and mean_de_cp_50!=0:
        comparison_df = pd.DataFrame({
            "Method": ["PPO", "CPOPT_1" ,"CPOPT_0","CPOPT_50","SPT","Gap_cp_1(%)", "Gap_cp_0(%)", "Gap_cp_50(%)", "Gap_SPT(%)" ],
            "Mean_ms": [mean_ms, mean_ms_cp_1,mean_ms_cp_0,mean_ms_cp_50,mean_ms_spt,((mean_ms-mean_ms_cp_1) / mean_ms_cp_1)*100,((mean_ms-mean_ms_cp_0) / mean_ms_cp_0)*100,((mean_ms-mean_ms_cp_50) / mean_ms_cp_50)*100,( (mean_ms-mean_ms_spt) / mean_ms_spt)*100],
            "Mean_deviation": [mean_de, mean_de_cp_1,mean_de_cp_0,mean_de_cp_50,mean_de_spt, ((obj_ppo-mean_de_cp_1) / mean_de_cp_1)*100,"-",((obj_ppo-mean_de_cp_50) / mean_de_cp_50)*100,((obj_ppo-obj_spt) / obj_spt)*100],
            "Mean_computational_time":[mean_time,mean_time_cp_1,mean_time_cp_0,mean_time_cp_50,mean_time_spt,"-","-","-","-"],
            "Objective_function":[obj_ppo ,obj_cp_1 ,obj_cp_0,obj_cp_50,obj_spt,"-","-","-","-"],
            "Initial_MS":[initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,"-","-"]
            
        })
    elif mean_de_cp_1!=0 and mean_de_cp_0!=0 and mean_de_cp_50==0:
        comparison_df = pd.DataFrame({
            "Method": ["PPO", "CPOPT_1" ,"CPOPT_0","CPOPT_50","SPT","Gap_cp_1(%)", "Gap_cp_0(%)", "Gap_cp_50(%)", "Gap_SPT(%)" ],
            "Mean_ms": [mean_ms, mean_ms_cp_1,mean_ms_cp_0,mean_ms_cp_50,mean_ms_spt,((mean_ms-mean_ms_cp_1) / mean_ms_cp_1)*100,((mean_ms-mean_ms_cp_0) / mean_ms_cp_0)*100,((mean_ms-mean_ms_cp_50) / mean_ms_cp_50)*100,( (mean_ms-mean_ms_spt) / mean_ms_spt)*100],
            "Mean_deviation": [mean_de, mean_de_cp_1,mean_de_cp_0,mean_de_cp_50,mean_de_spt, ((obj_ppo-mean_de_cp_1) / mean_de_cp_1)*100,((obj_ppo-mean_de_cp_0) / mean_de_cp_0)*100,"-",((obj_ppo-obj_spt) / obj_spt)*100],
            "Mean_computational_time":[mean_time,mean_time_cp_1,mean_time_cp_0,mean_time_cp_50,mean_time_spt,"-","-","-","-"],
            "Objective_function":[obj_ppo ,obj_cp_1 ,obj_cp_0,obj_cp_50,obj_spt,"-","-","-","-"],
            "Initial_MS":[initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,"-","-"]
            
        })

    filename = f"{inst}_{batch}.xlsx"
    filename = location+filename
    comparison_df.to_excel(filename, index=False)
if __name__ == "__main__":
    inst="10050"
    print("Comparison generated for instance:", inst)
    main(inst=inst, batch="b_1", arr_time="30")
    print("inst=inst, batch=b_1, arr_time=30: done.")
    main(inst=inst, batch="b_2", arr_time="30")
    print("inst=inst, batch=b_2, arr_time=30: done.")
    main(inst=inst, batch="b_3", arr_time="30")
    print("inst=inst, batch=b_3, arr_time=30: done.")
    main(inst=inst, batch="b_1", arr_time="60")
    print("inst=inst, batch=b_1, arr_time=60: done.")
    main(inst=inst, batch="b_2", arr_time="60")
    print("inst=inst, batch=b_2, arr_time=60: done.")
    main(inst=inst, batch="b_3", arr_time="60")
    print("inst=inst, batch=b_3, arr_time=60: done.")
    # main(inst=inst, batch="b_1", arr_time="90")
    # print("inst=inst, batch=b_1, arr_time=90: done.")
    # main(inst=inst, batch="b_2", arr_time="90")
    # print("inst=inst, batch=b_2, arr_time=90: done.")
    # main(inst=inst, batch="b_3", arr_time="90")
    # print("inst=inst, batch=b_3, arr_time=90: done.")

