import sys
# sys.path.append('../')
import time
import numpy as np
from docplex.cp.config import context
# context.solver.local.execfile = './x86-64_linux/cpoptimizer'
from docplex.cp.model import *
from obj import *
import pandas as pd
context.solver.log_output = None
def optimize(instance,hour,prev, dur , seq, start_dc, end, eliminated,max_time = 10000, time_limit = 100, threads = 1,alpha=1): 
    
    model = CpoModel('SimpleJobShop')
    interval_vars = dict()
    dev_vars = dict()
    # Create variables
    for task in instance.tasks:
        # print(task.length)
        interval_vars[task] = interval_var(start = (0, max_time), end = (0, max_time), size = task.length, name = 'interval'+str(task.name))
    for job in instance.jobs:
        # dev_vars[job]=integer_var(start=0, end=max_time, name='dev_job' + str(job.id))
        dev_vars[job]=integer_var(0, max_time, name='dev_job' + str(job.id))
    
    # Precedence and blocking constraints
    for task in instance.tasks:
        if task.next_task:
            model.add(start_of(interval_vars[task.next_task]) >= end_of(interval_vars[task]))
        
            
    # No overlap constraints
    for machine in instance.machines:
        machine_sequence = sequence_var([interval_vars[task] for task in machine.tasks])
        model.add(no_overlap(machine_sequence))
    
    eliminated_cp=[]
    # prev=0
    # Executed tasks constraint
    for job in instance.jobs:
        # print("job id, job pev:", job.id, job.prev_end,prev)

        if job.prev_end!=[]:    
            for task in job.tasks:        
                if (end[prev][job.id-1][int(task.order)-1]<=hour) or (end[prev][job.id-1][int(task.order)-1]>hour and start_dc[prev][job.id-1][int(task.order)-1]<hour ) :
                    eliminated_cp.append(task.name)
                    model.add(start_of(interval_vars[task])==int(start_dc[prev][job.id-1][int(task.order)-1]))
                    model.add(end_of(interval_vars[task])==int(end[prev][job.id-1][int(task.order)-1]))
            # print(eliminated_cp )        
            for task in job.tasks:
                if task.name not in eliminated_cp:
                    if hour>0:
                        model.add(start_of(interval_vars[task]) >= hour)

                
        else:
            for task in job.tasks:
                    if task.order==1:
                        model.add(start_of(interval_vars[task])>=hour)

            
   # deviation per job definition
    for job in instance.jobs:
        # if job.prev_end==[]:Here job.ref==0 means that it is a new job (15_04_2024)
        if job.ref==0:
            model.add(dev_vars[job]==0)
        else:
            # Create a new interval variable that represents the time difference between
            diff_interval = end_of(interval_vars[job.tasks[-1]]) - job.ref
            
        
        #Set the deviation variable to the maximum of 0 and the duration of the difference interval.
            model.add(dev_vars[job] == model.max(0, diff_interval))
       
    # Minimize the makespan
    obj_var1 = integer_var(0, max_time, 'makespan')
    obj_var2 = integer_var(0, max_time, 'dev')
    for task in instance.tasks:
        # model.add(obj_var1 >= end_of(interval_vars[task]))
        model.add(obj_var1 >= end_of(interval_vars[task]))
    model.add(obj_var2==sum(dev_vars.values()))
    # model.add(obj_var2<=100)
    model.minimize((1-alpha)*obj_var1+alpha*obj_var2)
    # Define solver and solve
    sol = model.solve(TimeLimit= time_limit, Workers = threads)
    
    solution = Solution(instance)
    start_cp=np.zeros_like(dur[hour], dtype=np.int32)
    end_cp=np.zeros_like(dur[hour], dtype=np.int32)
    # Print out solution and return it
    for job in instance.jobs:

        for task in job.tasks:
            start = sol.get_value(interval_vars[task])[0]
            end = sol.get_value(interval_vars[task])[1]
            start_cp[job.id-1, task.order-1]=start
            end_cp[job.id-1, task.order-1]=end
            solution.add(task, start, end)
    DE=sol.get_value(obj_var2)
    if alpha==1.0:
        MS=np.max(end_cp)
    else:
        MS=np.max(end_cp)
    sol_seq=[]

    df=[]
    data=[]
    for job in solution.instance.jobs:
        for task in job.tasks:
            sol_seq.append((task.name, sol.get_value(interval_vars[task]).start))
            if task.name in eliminated_cp:
                df.append(dict(Task='Task %s'%(task.name), Start=int(sol.get_value(interval_vars[task]).start), Finish=int(sol.get_value(interval_vars[task]).end), Resource= 'Mch %s'%(task.machine.id+1), Job='J %s'%(job.id), Completion=100))
            else:
                df.append(dict(Task='Task %s'%(task.name), Start=int(sol.get_value(interval_vars[task]).start), Finish=int(sol.get_value(interval_vars[task]).end), Resource= 'Mch %s'%(task.machine.id+1), Job='J %s'%(job.id), Completion=0))
           
    DF=pd.DataFrame(df)

    return solution, sol_seq,DF, data, eliminated_cp,start_cp,end_cp,DE,MS
 

def optimize_and_visualize(hour,prev,dur , seq, start_dc, end,eliminated, time_limit_x=0.3, threads=1, alpha=1):
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
        solution,sol_seq,DF, data, eliminated_cp,start_cp,end_cp,DE,MS = optimize(instance,hour,prev,dur , seq, start_dc, end,eliminated,max_time = 20000, time_limit=time_limit_x, threads=1,alpha=alpha)
        
        # MS=solution.get_makespan()
        t2=time.time()
        time_to=t1-t2
    return DF, data, eliminated_cp,solution,start_cp,end_cp,MS,DE,time_to
    
def main(
         inst:str="1010",
         batch:str="b_2",
         arr_time:str="30",
         alpha:float=1
         ):
  

    #Data initialization of the reference schedule
    location="./"
    # location="./"+inst+"/LG/SUM/"+batch+"/"+arr_time+"/"
    # location="./"+inst+"/LG/"+batch+"/"+arr_time+"/"
    print("location is:", location)
    simulations= open(location+"dico_data_multi.pkl", "rb")
    simulations=simulations.read()
    simulations=pickle.loads(simulations)



    # print(simulations[1]["compute_time"])
    dico_data_multi={}
    for simulation in simulations:
        # print("simulation is:", simulation)

        eliminated= simulations[simulation]["elimiminated_ops"]
        
        dur= simulations[simulation]["dur"]
        

        seq= simulations[simulation]["ops_seq"]

        start_dc= simulations[simulation]["op_start"]

        end= simulations[simulation]["End"]
        
        added= simulations[simulation]["added"]
        dico_compute_cp={}
        for hour in dur:
            # print("hour is:", hour)
        #Define the hour to make the comparison
            if hour==0:
                dico_dur_cp, dico_start_cp, dico_end_cp, dico_eliminated_cp, dico_ms_cp, dico_de_cp, dico_added={},{},{},{},{},{},{}
                prev=0
                cp_time=simulations[simulation]["compute_time"][hour]
                DF, data, eliminated_cp,sol_seq,start_cp,end_cp,MS,DE,time_to=optimize_and_visualize( hour,prev,dur , seq, start_dc, end,eliminated,cp_time,1,alpha=alpha)
                # print("end_cp0")
                # print(end_cp)
                # print("initial Ms:", MS,np.max(end_cp))
                # DF, data, eliminated_cp,sol_seq,start_cp,end_cp,MS,DE,time_to=optimize_and_visualize(hour,prev,dur , seq, start_dc, end,eliminated,1000,1)
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
                    # print("dico_end_cp",dico_end_cp[hour-1])

                    eliminated_cp= dico_eliminated_cp
                    # print("eliminated_cp",eliminated_cp)
                    
                    ms_cp= dico_ms_cp
                    # print("ms_cp",ms_cp)

                    de_cp= dico_de_cp
                    # print("de_cp",de_cp)
                    
                    dico_dur_cp, dico_start_cp, dico_end_cp, dico_eliminated_cp, dico_ms_cp, dico_de_cp = dur_cp, start_dc, end_cp, eliminated_cp, ms_cp,de_cp
                    prev=list(end_cp.keys())[-1]
                    cp_time=simulations[simulation]["compute_time"][hour]
                    # print("cp_time",cp_time)
                    # print("initial end", len(end_cp),end_cp[hour-1])
                    DF, data, eliminated_cp,solution,start_cp,end_cp,MS,DE,time_to=optimize_and_visualize( hour,prev,dur , seq, start_dc, end_cp,eliminated, cp_time, 1,alpha=alpha)
                    # print("end_cp1")
                    # print(end_cp)
                    # print("MS, DE:",np.max(end_cp),MS,DE)
                    
                    # print("DE is", DE)
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

            dico_ms_cp[hour]= MS
            dico_de_cp[hour]=DE
            dico_dur_cp[hour]=dur[hour]   
            dico_start_cp[hour]=start_cp
            dico_end_cp[hour]=end_cp
            dico_eliminated_cp[hour]=eliminated_cp
        

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
    # scenarios_file=open("cplex_data.pkl", "wb")
    if alpha==1:
        scenarios_file=open(location+"cplex_data_1.pkl", "wb")
        pickle.dump(dico_data_multi, scenarios_file)
        scenarios_file.close()
    elif alpha==0:
        scenarios_file=open(location+"cplex_data_0.pkl", "wb")
        pickle.dump(dico_data_multi, scenarios_file)
        scenarios_file.close()
    else:
        scenarios_file=open(location+"cplex_data_50.pkl", "wb")
        pickle.dump(dico_data_multi, scenarios_file)
        scenarios_file.close()
    
    # print("calculation is over!")
    return
if __name__ == "__main__":
    inst="1515"
    alpha=1.0
    print("cplex solution generation for inst:", inst,"alpha=",alpha)
    main(inst=inst, batch="b_1", arr_time="30",alpha=alpha)
    print("inst=inst, batch=b_1, arr_time=30: done.")
    main(inst=inst, batch="b_2", arr_time="30",alpha=alpha)
    print("inst=inst, batch=b_2, arr_time=30: done.")
    main(inst=inst, batch="b_3", arr_time="30",alpha=alpha)
    print("inst=inst, batch=b_3, arr_time=30: done.")
    main(inst=inst, batch="b_1", arr_time="60",alpha=alpha)
    print("inst=inst, batch=b_1, arr_time=60: done.")
    main(inst=inst, batch="b_2", arr_time="60",alpha=alpha)
    print("inst=inst, batch=b_2, arr_time=60: done.")
    main(inst=inst, batch="b_3", arr_time="60",alpha=alpha)
    print("inst=inst, batch=b_3, arr_time=60: done.")
    main(inst=inst, batch="b_1", arr_time="90",alpha=alpha)
    print("inst=inst, batch=b_1, arr_time=90: done.")
    main(inst=inst, batch="b_2", arr_time="90",alpha=alpha)
    print("inst=inst, batch=b_2, arr_time=90: done.")
    main(inst=inst, batch="b_3", arr_time="90",alpha=alpha)
    print("inst=inst, batch=b_3, arr_time=90: done.")

    # alpha=0: MS_only$
    alpha=0.0
    print("cplex solution generation for inst:", inst,"alpha=",alpha)
    main(inst=inst, batch="b_1", arr_time="30",alpha=alpha)
    print("inst=inst, batch=b_1, arr_time=30: done.")
    main(inst=inst, batch="b_2", arr_time="30",alpha=alpha)
    print("inst=inst, batch=b_2, arr_time=30: done, alpha:", alpha)
    main(inst=inst, batch="b_3", arr_time="30",alpha=alpha)
    print("inst=inst, batch=b_3, arr_time=30: done.")
    main(inst=inst, batch="b_1", arr_time="60",alpha=alpha)
    print("inst=inst, batch=b_1, arr_time=60: done.")
    main(inst=inst, batch="b_2", arr_time="60",alpha=alpha)
    print("inst=inst, batch=b_2, arr_time=60: done.")
    main(inst=inst, batch="b_3", arr_time="60",alpha=alpha)
    print("inst=inst, batch=b_3, arr_time=60: done.")
    # main(inst=inst, batch="b_1", arr_time="90",alpha=alpha)
    # print("inst=inst, batch=b_1, arr_time=90: done.")
    # main(inst=inst, batch="b_2", arr_time="90",alpha=alpha)
    # print("inst=inst, batch=b_2, arr_time=90: done.")
    # main(inst=inst, batch="b_3", arr_time="90",alpha=alpha)
    # print("inst=inst, batch=b_3, arr_time=90: done.")

    # # alpha=50%
    alpha=0.75
    print("cplex solution generation for inst:", inst,"alpha=",alpha)
    main(inst=inst, batch="b_1", arr_time="30",alpha=alpha)
    print("inst=inst, batch=b_1, arr_time=30: done.")
    main(inst=inst, batch="b_2", arr_time="30",alpha=alpha)
    print("inst=inst, batch=b_2, arr_time=30: done.")
    main(inst=inst, batch="b_3", arr_time="30",alpha=alpha)
    print("inst=inst, batch=b_3, arr_time=30: done.")
    main(inst=inst, batch="b_1", arr_time="60",alpha=alpha)
    print("inst=inst, batch=b_1, arr_time=60: done.")
    main(inst=inst, batch="b_2", arr_time="60",alpha=alpha)
    print("inst=inst, batch=b_2, arr_time=60: done.")
    main(inst=inst, batch="b_3", arr_time="60",alpha=alpha)
    print("inst=inst, batch=b_3, arr_time=60: done.")
    # main(inst=inst, batch="b_1", arr_time="90",alpha=alpha)
    # print("inst=inst, batch=b_1, arr_time=90: done.")
    # main(inst=inst, batch="b_2", arr_time="90",alpha=alpha)
    # print("inst=inst, batch=b_2, arr_time=90: done.")
    # main(inst=inst, batch="b_3", arr_time="90",alpha=alpha)
    # print("inst=inst, batch=b_3, arr_time=90: done.")
    
    