import pandas as pd
import pickle
# inst = "1515"
# alpha= "0.75"
# arr_time="30"
# batch = "b_2"

# Defiene extraction function
def extract(data):
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
    obj_ppo=0.75* mean_de + 0.25*mean_ms

    return initial_ms,mean_ms, mean_de,mean_time,obj_ppo


def main(
         inst:str="1010",
         batch:str="b_2",
         arr_time:str="30",
         ):
    location="./"+inst+"/NOLG/"+batch+"/"+arr_time+"/"
    # location="./"+inst+"/0.75/"+batch+"/"
    # lc="./"+inst+"/NOLG/"+batch+"/"+arr_time+"/"
    print("location is:", location)
    # import matplotlib.pyplot as plt
    # Assuming your PPO dictionary of simulations is named "data"
    data = open(location+"dico_data_multi.pkl", "rb")
    data = data.read()
    data = pickle.loads(data)

    # Assuming your CPLEX dictionary of simulations is named "cplex_data"
    # cplex_data = open("cplex_data.pkl", "rb")
    cplex_data = open(location+"cplex_data.pkl", "rb")
    cplex_data = cplex_data.read()
    cplex_data = pickle.loads(cplex_data)

    # mip_data = open(location+"MIP_data.pkl", "rb")
    mip_data = open(location+"MIP_data.pkl", "rb")
    mip_data = mip_data.read()
    mip_data = pickle.loads(mip_data)

    LPT_data = open(lc+"LPT_data.pkl", "rb")
    LPT_data = LPT_data.read()
    LPT_data = pickle.loads(LPT_data)

    SPT_data = open(lc+"SPT_data.pkl", "rb")
    SPT_data = SPT_data.read()
    SPT_data = pickle.loads(SPT_data)

    FIFO_data = open(lc+"FIFO_data.pkl", "rb")
    FIFO_data = FIFO_data.read()
    FIFO_data = pickle.loads(FIFO_data)

    LIFO_data = open(lc+"LIFO_data.pkl", "rb")
    LIFO_data = LIFO_data.read()
    LIFO_data = pickle.loads(LIFO_data)

    initial_ms,mean_ms, mean_de,mean_time,obj_ppo=extract(data)

    initial_ms,mean_ms_cp, mean_de_cp,mean_time_cp,obj_cp=extract(cplex_data)

    initial_ms,mean_ms_mip, mean_de_mip,mean_time_mip,obj_mip=extract(mip_data)

    initial_ms,mean_ms_lpt, mean_de_lpt,mean_time_lpt,obj_lpt=extract(LPT_data)

    initial_ms,mean_ms_spt, mean_de_spt,mean_time_spt,obj_spt=extract(SPT_data)

    initial_ms,mean_ms_fifo, mean_de_fifo,mean_time_fifo,obj_fifo=extract(FIFO_data)

    initial_ms,mean_ms_lifo, mean_de_lifo,mean_time_lifo,obj_lifo=extract(LIFO_data)

    print("initialms is:", initial_ms)
    comparison_df = pd.DataFrame({
        "Method": ["PPO", "CPLEX" ,"MIP","LPT","SPT","FIFO", "LIFO","Gap_cp(%)", "Gap_mip(%)", "Gap_LPT(%)", "Gap_SPT(%)","Gap_FIFO(%)","Gap_LIFO(%)" ],
        "Mean_ms": [mean_ms, mean_ms_cp,mean_ms_mip,mean_ms_lpt,mean_ms_spt,mean_ms_fifo,mean_ms_lifo,((mean_ms-mean_ms_cp) / mean_ms_cp)*100,( (mean_ms-mean_ms_mip) / mean_ms_mip)*100,( (mean_ms-mean_ms_lpt) / mean_ms_lpt)*100,( (mean_ms-mean_ms_spt) / mean_ms_spt)*100,( (mean_ms-mean_ms_fifo) / mean_ms_fifo)*100,( (mean_ms-mean_ms_lifo) / mean_ms_lifo)*100],
        "Mean_deviation": [mean_de, mean_de_cp,mean_de_mip,mean_de_lpt,mean_de_spt,mean_de_fifo,mean_de_lifo, "-","-","-","-","-","-"],
        "Mean_computational_time":[mean_time,mean_time_cp,mean_time_mip,mean_time_lpt,mean_time_spt,mean_time_fifo,mean_time_lifo,"-","-","-","-","-","-"],
        "Objective_function":[obj_ppo ,obj_cp ,obj_mip,obj_lpt,obj_spt,obj_fifo,obj_lifo,((obj_ppo-obj_cp) / obj_cp)*100,((obj_ppo-obj_mip) / obj_mip)*100,((obj_ppo-obj_lpt) / obj_lpt)*100,((obj_ppo-obj_spt) / obj_spt)*100,((obj_ppo-obj_fifo) / obj_fifo)*100,((obj_ppo-obj_lifo) / obj_lifo)*100],
        "Initial_MS":[initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,initial_ms,"-","-","-","-","-","-"]
        
    })
    # "( (mean_de-mean_de_lpt) / mean_de_lpt)*100"

    filename = f"{inst}_{batch}.xlsx"
    filename = location+filename
    comparison_df.to_excel(filename, index=False)
if __name__ == "__main__":
    inst="2020"
    # print("Comparison generated for instance:", inst)
    # main(inst=inst, batch="b_1", arr_time="30")
    # print("inst=inst, batch=b_1, arr_time=30: done.")
    # main(inst=inst, batch="b_2", arr_time="30")
    # print("inst=inst, batch=b_2, arr_time=30: done.")
    # main(inst=inst, batch="b_3", arr_time="30")
    # print("inst=inst, batch=b_3, arr_time=30: done.")
    # main(inst=inst, batch="b_1", arr_time="60")
    # print("inst=inst, batch=b_1, arr_time=60: done.")
    # main(inst=inst, batch="b_2", arr_time="60")
    # print("inst=inst, batch=b_2, arr_time=60: done.")
    # main(inst=inst, batch="b_3", arr_time="60")
    # print("inst=inst, batch=b_3, arr_time=60: done.")
    main(inst=inst, batch="b_1", arr_time="90")
    print("inst=inst, batch=b_1, arr_time=90: done.")
    # main(inst=inst, batch="b_2", arr_time="90")
    # print("inst=inst, batch=b_2, arr_time=90: done.")
    # main(inst=inst, batch="b_3", arr_time="90")
    # print("inst=inst, batch=b_3, arr_time=90: done.")

