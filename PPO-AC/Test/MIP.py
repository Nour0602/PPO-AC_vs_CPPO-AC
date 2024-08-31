import sys
import time
import numpy as np
from docplex.mp.model import *
from obj import *
# import plotly.express as px
import pandas as pd
import logging 

logging.getLogger('docplex.mp').setLevel(logging.ERROR)


def optimize(instance,hour,prev, dur , seq, start_dc, end, eliminated,max_time = 10000, time_limit = 100, threads = 1): 
    model = Model('SimpleJobShop')
    
    start_time = dict()
    end_time = dict()
    dev_vars = dict()
    bigm = max_time

    # Create variables
    for task in instance.tasks:
        start_time[task] = model.continuous_var()
        end_time[task] = model.continuous_var()

    for job in instance.jobs:
        # dev_vars[job]=integer_var(start=0, end=max_time, name='dev_job' + str(job.id))
        dev_vars[job]=model.continuous_var()

    # Precedence and blocking constraints
    for task in instance.tasks:
        model.add(end_time[task]==start_time[task]+task.length)
        if task.next_task:
            model.add(start_time[task.next_task] >= task.length+start_time[task])

    # No overlap constraints
    for machine in instance.machines:
        for t1 in machine.tasks:
            for t2 in machine.tasks:
                if t1.name > t2.name:
                    prec = model.binary_var(name = t1.name+'_precedes_'+t2.name)
                    model.add(start_time[t1] + t1.length - bigm*(1-prec) <= start_time[t2])
                    model.add(start_time[t2] + t2.length - bigm*prec <= start_time[t1])
    

    
    eliminated_cp=[]
    # prev=0
    # Executed tasks constraint
    for job in instance.jobs:
        # print("job id, job pev:", job.id, job.prev_end,prev)

        if job.prev_end!=[]:    
            for task in job.tasks:        
                if (end[prev][job.id-1][int(task.order)-1]<=hour) or (end[prev][job.id-1][int(task.order)-1]>hour and start_dc[prev][job.id-1][int(task.order)-1]<hour ) :
                    eliminated_cp.append(task.name)
                    model.add(start_time[task]==int(start_dc[prev][job.id-1][int(task.order)-1]))
                    model.add(end_time[task]==int(end[prev][job.id-1][int(task.order)-1]))
            # print(eliminated_cp )        
            for task in job.tasks:
                if task.name not in eliminated_cp:
                    if hour>0:
                        model.add(start_time[task] >= hour)

                
        else:
            for task in job.tasks:
                    if task.order==1:
                        model.add(start_time[task]>=hour)

            
   # deviation per job definition
    for job in instance.jobs:
        # if job.prev_end==[]:
        if job.ref==0:
            model.add(dev_vars[job]==0)
        else:
            # Create a new interval variable that represents the time difference between
            diff_interval = end_time[job.tasks[-1]] - job.ref
            
        
        # Set the deviation variable to the maximum of 0 and the duration of the difference interval.
            model.add(dev_vars[job] == model.max(0, diff_interval))
       
    # Minimize the makespan
    obj_var1 = model.continuous_var(0, max_time, 'makespan')
    obj_var2 = model.continuous_var(0,max_time, 'dev')

    for task in instance.tasks:
        model.add(obj_var1 >= end_time[task] )
    # for job in instance.jobs:
    #     model.add(obj_var2 >=dev_vars[job])
    model.add(obj_var2==sum(dev_vars.values()))
    # model.add(obj_var2<=100)
    # model.minimize(0.25*obj_var1+0.75*obj_var2)
    model.minimize(0.25*obj_var1+0.75*obj_var2)
    # model.minimize(obj_var1)
    # model.minimize(obj_var2)
    
    # Define solver and solve
    model.parameters.timelimit.set(time_limit)
    model.parameters.threads.set(threads)
    sol = model.solve(log_output = True)

    solution = Solution(instance)

    start_cp=np.zeros_like(dur[hour], dtype=np.int32)
    end_cp=np.zeros_like(dur[hour], dtype=np.int32)
    # print("end_cp zeros:", end_cp)
    # Print out solution and return it
    for job in instance.jobs:

        for task in job.tasks:
            # print(task.name, task.machine.id)
            start = sol.get_value(start_time[task])
            end=sol.get_value(end_time[task])
            # print("end is:", end)
            start_cp[job.id-1, task.order-1]=start
            end_cp[job.id-1, task.order-1]=end
            
            
            # print('Start: %f'%start)
            # print('End: %f'%end)
            solution.add(task, start, end)
    DE=sol.get_value(obj_var2)
    sol_seq=[]

    df=[]
    data=[]
    for job in solution.instance.jobs:
        for task in job.tasks:
            sol_seq.append((task.name, sol.get_value(start_time[task])))
            if task.name in eliminated_cp:
                df.append(dict(Task='Task %s'%(task.name), Start=int(sol.get_value(start_time[task])), Finish=int(sol.get_value(end_time[task])), Resource= 'Mch %s'%(task.machine.id+1), Job='J %s'%(job.id), Completion=100))
            else:
                df.append(dict(Task='Task %s'%(task.name), Start=int(sol.get_value(start_time[task])), Finish=int(sol.get_value(end_time[task])), Resource= 'Mch %s'%(task.machine.id+1), Job='J %s'%(job.id), Completion=0))
           
    DF=pd.DataFrame(df)

    print("solution status:", model.solve_details.status)
    
    
    return solution, sol_seq,DF, data, eliminated_cp,start_cp,end_cp,DE

def optimize_and_visualize(hour,prev,dur , seq, start_dc, end,eliminated, time_limit_x=0.3, threads=1):
    t1=time.time()
    reader = reading(hour,prev,dur , seq, start_dc, end, eliminated)
    instance = reader.get_instance()
    if hour==0:
        df=[]
        data=[]
        for job in instance.jobs:
            for task in job.tasks:
                df.append(dict(Task='Task %s'%(task.name), Start=start_dc[hour][job.id-1][task.order-1], Finish=end[hour][job.id-1][task.order-1], Resource= 'Mch %s'%(task.machine.id+1), Job='J %s'%(job.id), Completion=0))
        DF=pd.DataFrame(df)    

        sol_seq, eliminated_cp,start_cp,end_cp,DE=seq[hour],eliminated[hour], start_dc[hour], end[hour],0
        MS=end[hour].max()
        t2=time.time()
        time_to=t1-t2
        return DF, data, eliminated_cp,sol_seq,start_cp,end_cp,MS,DE,time_to
    
    else:
        # print("prev_optim_visualize", prev)
        solution,sol_seq,DF, data, eliminated_cp,start_cp,end_cp,DE = optimize(instance,hour,prev,dur , seq, start_dc, end,eliminated,max_time = 10000, time_limit=time_limit_x, threads=1)
        
        MS=solution.get_makespan()
        t2=time.time()
        time_to=t1-t2
    return DF, data, eliminated_cp,solution,start_cp,end_cp,MS,DE,time_to
def main(
         inst:str="1010",
         batch:str="b_2",
         arr_time:str="30",
         ):
    #Data initialization of the reference schedule
    
    # location="./"+inst+"/NOLG/"+batch+"/"+arr_time+"/"
    location="./"
    print("location:",location)

    simulations= open(location+"dico_data_multi.pkl", "rb")
    simulations=simulations.read()
    simulations=pickle.loads(simulations)



    # print(simulations[1]["compute_time"])
    dico_data_multi={}
    for simulation in simulations:
        print("simulation is:", simulation)


        eliminated= simulations[simulation]["elimiminated_ops"]
        
        dur= simulations[simulation]["dur"]
        

        seq= simulations[simulation]["ops_seq"]

        start_dc= simulations[simulation]["op_start"]

        end= simulations[simulation]["End"]
        
        added= simulations[simulation]["added"]
        dico_compute_cp={}
        for hour in dur:
        #Define the hour to make the comparison
            if hour==0:
                dico_dur_cp, dico_start_cp, dico_end_cp, dico_eliminated_cp, dico_ms_cp, dico_de_cp, dico_added={},{},{},{},{},{},{}
                prev=0
                cp_time=simulations[simulation]["compute_time"][hour]
                DF, data, eliminated_cp,sol_seq,start_cp,end_cp,MS,DE,time_to=optimize_and_visualize( hour,prev,dur , seq, start_dc, end,eliminated,cp_time,1)
                # DF, data, eliminated_cp,sol_seq,start_cp,end_cp,MS,DE,time_to=optimize_and_visualize( hour,prev,dur , seq, start_dc, end,eliminated,1000,1)
                dico_compute_cp[hour]=0
                dico_ms_cp[hour]= MS
                dico_de_cp[hour]=DE
                dico_dur_cp[hour]=dur[hour]   
                dico_start_cp[hour]=start_cp
                dico_end_cp[hour]=end_cp
                dico_eliminated_cp[hour]=eliminated_cp
            else:

                if simulations[simulation]["added"][hour]!=0:
                    dur_cp=dico_dur_cp
                    # print("dur_cp", dur_cp)

                    start_dc= dico_start_cp

                    end_cp = dico_end_cp

                    eliminated_cp= dico_eliminated_cp
                    # print("eliminated_cp",eliminated_cp)
                    
                    ms_cp= dico_ms_cp
                    # print("ms_cp",ms_cp)

                    de_cp= dico_de_cp
                    # print("de_cp",de_cp)
                    



                    dico_dur_cp, dico_start_cp, dico_end_cp, dico_eliminated_cp, dico_ms_cp, dico_de_cp = dur_cp, start_dc, end_cp, eliminated_cp, ms_cp,de_cp
                    prev=list(end_cp.keys())[-1]
                    cp_time=simulations[simulation]["compute_time"][hour]
                    print("cp_time",cp_time)
                    DF, data, eliminated_cp,solution,start_cp,end_cp,MS,DE,time_to=optimize_and_visualize( hour,prev,dur , seq, start_dc, end_cp,eliminated,3600 , 2)
                    # DF, data, eliminated_cp,sol_seq,start_cp,end_cp,MS,DE,time_to=optimize_and_visualize( hour,prev,dur , seq, start_dc, end,eliminated,1000,1)
                    dico_compute_cp[hour]=time_to*(-1)
                    dico_ms_cp[hour]= MS
                    dico_de_cp[hour]=DE
                    dico_dur_cp[hour]=dur[hour]   
                    dico_start_cp[hour]=start_cp
                    dico_end_cp[hour]=end_cp
                    dico_eliminated_cp[hour]=eliminated_cp

                    
                else:
                    dico_ms_cp[hour]= dico_ms_cp[hour-1]
                    dico_de_cp[hour]=dico_de_cp[hour-1]
                    dico_dur_cp[hour]=dico_dur_cp[hour-1]   
                    dico_start_cp[hour]=dico_start_cp[hour-1]
                    dico_end_cp[hour]=dico_end_cp[hour-1]
                    dico_eliminated_cp[hour]=dico_eliminated_cp[hour-1]
                    dico_compute_cp[hour]=0

        

        dico_sim={}
        dico_sim["Ms"]=dico_ms_cp
        dico_sim["DE"]=dico_de_cp
        dico_sim["compute_time"]=dico_compute_cp

        dico_sim["elimiminated_ops"]=dico_eliminated_cp
        dico_sim["dur"]=dico_dur_cp
        dico_sim["op_start"]=dico_start_cp
        dico_sim["End"]=dico_end_cp 
        dico_sim["added"]=added

        dico_data_multi[simulation]=dico_sim
    scenarios_file=open(location+"MIP_data.pkl", "wb")
    pickle.dump(dico_data_multi, scenarios_file)
    scenarios_file.close() 
    print("calculation is over!")
    return
if __name__ == "__main__":
    inst="2020"
    # print("MIP solution generation for inst:", inst)
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
  
    
 
    
    

