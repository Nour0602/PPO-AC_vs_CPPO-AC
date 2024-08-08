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
    location="./"+inst+"/General/small_large/"+arr_time+"/"
    lc_orig="./"+inst+"/NOLG/"+batch+"/"+arr_time+"/"
    print("location is:", location)
    # import matplotlib.pyplot as plt
    # Assuming your PPO dictionary of simulations is named "data"
    data = open(location+"dico_data_multi.pkl", "rb")
    data = data.read()
    data = pickle.loads(data)

    data_orig = open(lc_orig+"dico_data_multi.pkl", "rb")
    data_orig = data_orig.read()
    data_orig = pickle.loads(data_orig)


    initial_ms,mean_ms, mean_de,mean_time,obj_ppo=extract(data)
    initial_ms_orig,mean_ms_orig, mean_de_orig,mean_time_orig,obj_ppo_orig=extract(data_orig)

   

    print("initialms is:", initial_ms)
    comparison_df = pd.DataFrame({
        "Method": ["PPO", "PPO_Orig","Gap_cp(%)" ],
        "Mean_ms": [mean_ms, mean_ms_orig,((mean_ms-mean_ms_orig) / mean_ms_orig)*100 ],
        "Mean_deviation": [mean_de, mean_de_orig,"-",],
        "Mean_computational_time":[mean_time,mean_time_orig,"-"],
        "Objective_function":[obj_ppo ,obj_ppo_orig ,((obj_ppo-obj_ppo_orig) / obj_ppo_orig)*100],
        "Initial_MS":[initial_ms,initial_ms,"-"]
        
    })
    # "( (mean_de-mean_de_lpt) / mean_de_lpt)*100"

    filename = f"{inst}_{batch}.xlsx"
    filename = location+filename
    comparison_df.to_excel(filename, index=False)
if __name__ == "__main__":
    inst="3020"
    print("Comparison generated for instance:", inst)
    main(inst=inst, batch="b_1", arr_time="30")
    print("inst=inst, batch=b_1, arr_time=30: done.")
    # main(inst=inst, batch="b_2", arr_time="30")
    # print("inst=inst, batch=b_2, arr_time=30: done.")
    # main(inst=inst, batch="b_3", arr_time="30")
    # print("inst=inst, batch=b_3, arr_time=30: done.")
    main(inst=inst, batch="b_1", arr_time="60")
    print("inst=inst, batch=b_1, arr_time=60: done.")
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

