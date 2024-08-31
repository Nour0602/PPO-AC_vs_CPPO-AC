import copy
import pickle
import sys
import time
import numpy as np
from obj import *
from CPOPTIMIZER import optimize_and_visualize

def find_earliest_time(schedule, machine,job):
    times_on_machine = [t[2]+t[0].length for t in schedule if t[0].machine == machine] 
   
    prev_task = [t[2]+t[0].length for t in schedule if t[0].job == job ]  
       
    if not times_on_machine:
        MCH=0
    else:
        MCH=max(times_on_machine)
    if not prev_task:
        JCH=0
    else:
        JCH=max(prev_task)
    
    return max(MCH,JCH)

def schedule_LPT(instance,end,hour,start_dc,prev):
    eliminated_cp=[]
# Executed tasks constraint
    for job in instance.jobs:
        if job.prev_end!=[]:    
            for task in job.tasks:        
                if (end[prev][job.id-1][int(task.order)-1]<=hour) or (end[prev][job.id-1][int(task.order)-1]>hour and start_dc[prev][job.id-1][int(task.order)-1]<hour ) :
                    eliminated_cp.append(task.name)
    
    task_list= []
    processed=[]
    for job in instance.jobs:
        for task in job.tasks:
            if task.name not in eliminated_cp:
                task_list.append((task,job.id))
            else:
                processed.append((task,job.id,start_dc[prev][job.id-1][task.order-1], task.name))
    # print("Task list is:", task_list) 
# task_list = [(task1, job1), (task2, job2), (task3, job1)] 

    job_task_lengths = {}

    for task, job in task_list:
        if job not in job_task_lengths:
            job_task_lengths[job] = 0
    
        job_task_lengths[job] += task.length

# job_task_lengths now contains the sum of lengths 
# for each job
    sorted_jobs = sorted(job_task_lengths, key=job_task_lengths.get, reverse=True) 
    
    unscheduled_tasks = []
    for job in sorted_jobs:
        for task, orig_job in task_list:
            if orig_job == job:
                unscheduled_tasks.append((task, orig_job))

    # print("unscheduled tasks:", unscheduled_tasks)

    scheduled_tasks = processed

    while unscheduled_tasks:
        
        task = unscheduled_tasks.pop(0)[0]
        machine = task.machine
        job=task.job
        start_time = find_earliest_time(scheduled_tasks, machine,job)
     
        scheduled_tasks.append((task,task.job.id,start_time,task.name))
       

    return scheduled_tasks


def schedule_SPT(instance,end,hour,start_dc,prev):

    # print("start dc", start_dc[prev])
    eliminated_cp=[]
# Executed tasks constraint
    for job in instance.jobs:
        # print("eliminated cp is:", eliminated_cp)
        if job.prev_end!=[]:    
            for task in job.tasks:        
                if (end[prev][job.id-1][int(task.order)-1]<=hour) or (end[prev][job.id-1][int(task.order)-1]>hour and start_dc[prev][job.id-1][int(task.order)-1]<hour ) :
                    eliminated_cp.append(task.name)
    

    task_list= []
    processed=[]
    for job in instance.jobs:
        for task in job.tasks:
            if task.name not in eliminated_cp:
                task_list.append((task,job.id))
            else:
                processed.append((task,job.id,start_dc[prev][job.id-1][task.order-1], task.name))
     

    job_task_lengths = {}

    for task, job in task_list:
        if job not in job_task_lengths:
            job_task_lengths[job] = 0
    
        job_task_lengths[job] += task.length

# job_task_lengths now contains the sum of lengths 
    sorted_jobs = sorted(job_task_lengths, key=job_task_lengths.get, reverse=False) 
    
    unscheduled_tasks = []
    for job in sorted_jobs:
        for task, orig_job in task_list:
            if orig_job == job:
                unscheduled_tasks.append((task, orig_job))

    scheduled_tasks = processed

    while unscheduled_tasks:
        
        task = unscheduled_tasks.pop(0)[0]
        machine = task.machine
        job=task.job
        start_time = find_earliest_time(scheduled_tasks, machine,job)
     
        scheduled_tasks.append((task,task.job.id,start_time,task.name))
       

    return scheduled_tasks

def schedule_FIFO(instance,end,hour,start_dc,prev):

    eliminated_cp=[]
# Executed tasks constraint
    for job in instance.jobs:
        if job.prev_end!=[]:    
            for task in job.tasks:        
                if (end[prev][job.id-1][int(task.order)-1]<=hour) or (end[prev][job.id-1][int(task.order)-1]>hour and start_dc[prev][job.id-1][int(task.order)-1]<hour ) :
                    eliminated_cp.append(task.name)
    
    task_list= []
    processed=[]
    for job in instance.jobs:
        for task in job.tasks:
            if task.name not in eliminated_cp:
                task_list.append((task,job.id))
            else:
                processed.append((task,job.id,start_dc[prev][job.id-1][task.order-1], task.name))
  
    unscheduled_tasks = []
    
    for task, orig_job in task_list:
        
        unscheduled_tasks.append((task, orig_job))

    scheduled_tasks = processed

    while unscheduled_tasks:
        
        task = unscheduled_tasks.pop(0)[0]
        machine = task.machine
        job=task.job
        start_time = find_earliest_time(scheduled_tasks, machine,job)
     
        scheduled_tasks.append((task,task.job.id,start_time,task.name))
       

    return scheduled_tasks


def schedule_LIFO(instance,end,hour,start_dc,prev):
    eliminated_cp=[]
# Executed tasks constraint
    for job in instance.jobs:
        if job.prev_end!=[]:    
            for task in job.tasks:        
                if (end[prev][job.id-1][int(task.order)-1]<=hour) or (end[prev][job.id-1][int(task.order)-1]>hour and start_dc[prev][job.id-1][int(task.order)-1]<hour ) :
                    eliminated_cp.append(task.name)
    
    task_list= []
    processed=[]
    for job in instance.jobs:
        for task in job.tasks:
            if task.name not in eliminated_cp:
                task_list.append((task,job.id))
            else:
                processed.append((task,job.id,start_dc[prev][job.id-1][task.order-1], task.name))

    # Get unique job ids
    job_ids = []
    for task, job_id in task_list:
        if job_id not in job_ids:
            job_ids.append(job_id)

    # Map job_id to list of tasks 
    job_tasks = {}
    for task, job_id in task_list:
        if job_id not in job_tasks:
            job_tasks[job_id] = []
    
        job_tasks[job_id].append(task)

    # Construct LIFO order  
    unscheduled_tasks = []
    for job_id in reversed(sorted(job_ids)):
        for task in job_tasks[job_id]:
            unscheduled_tasks.append((task,task.job.id))

    scheduled_tasks = processed

    while unscheduled_tasks:
        
        task = unscheduled_tasks.pop(0)[0]
        machine = task.machine
        job=task.job
        start_time = find_earliest_time(scheduled_tasks, machine,job)
     
        scheduled_tasks.append((task,task.job.id,start_time,task.name))
       

    return scheduled_tasks

def apply(heuristic:callable, data_file:str,inst:str="1010",
         batch:str="b_2",
         arr_time:str="30",)->None:
    #Data initialization of the reference schedule
    location="./"+inst+"/LG/"+batch+"/"+arr_time+"/"
    lc="./"
    # simulations= open(location+"dico_data_multi.pkl", "rb")
    simulations= open(lc+"dico_data_multi.pkl", "rb")
    simulations=simulations.read()
    simulations=pickle.loads(simulations)
    dico_data_multi={}
    for simulation in simulations:

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
                # print("hour is:", hour)
                dico_dur_cp, dico_start_cp, dico_end_cp, dico_eliminated_cp, dico_ms_cp, dico_de_cp, dico_added={},{},{},{},{},{},{}
                prev=0
                cp_time=simulations[simulation]["compute_time"][hour]
                DF, data, eliminated_cp,sol_seq,start_cp,end_cp,MS,DE,time_to=optimize_and_visualize( hour,prev,dur , seq, start_dc, end,eliminated,cp_time,100)
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
                    
                    start_dc= dico_start_cp

                    end_cp = dico_end_cp

                    eliminated_cp= dico_eliminated_cp
                    
                    ms_cp= dico_ms_cp
                    
                    de_cp= dico_de_cp
                   
                    dico_dur_cp, dico_start_cp, dico_end_cp, dico_eliminated_cp, dico_ms_cp, dico_de_cp = dur_cp, start_dc, end_cp, eliminated_cp, ms_cp,de_cp
                    prev=list(end_cp.keys())[-1]
                    cp_time=simulations[simulation]["compute_time"][hour]
            
                    reader = reading(hour,prev,dur , seq, start_dc, end, eliminated)
                    instance = reader.get_instance()
                    t1=time.time()

                    scheduled_tasks= heuristic(instance, end_cp, hour, start_dc,prev)
                    t2=time.time()
                    time_to=t1-t2
                    job_ref=[]
                    job_endg=[]
                    deviations = []
                    makespan = 0
                    for task,job_id, start_time, name in scheduled_tasks:
                        end_time = start_time + task.length
                        makespan = max(makespan, end_time)
       
                    for job in instance.jobs:
                        if job.ref!=0:
                            L=[t[2]+t[0].length for t in scheduled_tasks if t[0].job == job]
                            job_end= max(L)
                            deviations.append(max(0,(job_end - job.ref)))
                            job_ref.append(job.ref)     
                            job_endg.append(job_end)
                        else:
                            L=[t[2]+t[0].length for t in scheduled_tasks if t[0].job == job]
                            job_end= max(L)
                            deviations.append(0)
                            job_ref.append(0)
                            job_endg.append(job_end)

                    DE= sum(deviations)
                    MS= makespan
                    print("heuristic:", heuristic)
                    print("job_ref, job_endg are:", job_ref, job_endg)
                    print("Dev, deviationsa are:", MS,DE,deviations)
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

    # scenarios_file=open(location+f"./{data_file}.pkl", "wb")
    scenarios_file=open(lc+f"./{data_file}.pkl", "wb")
    pickle.dump(dico_data_multi, scenarios_file)
    scenarios_file.close() 
    
            
def main(inst:str="1010",
         batch:str="b_2",
         arr_time:str="30"):
    # apply(heuristic=schedule_LPT, data_file="LPT_data",inst=inst,
    #      batch=batch,
    #      arr_time=arr_time)
    # apply(heuristic=schedule_FIFO, data_file="FIFO_data",inst=inst,
    #      batch=batch,
    #      arr_time=arr_time)
    # apply(heuristic=schedule_LIFO, data_file="LIFO_data",inst=inst,
    #      batch=batch,
    #      arr_time=arr_time)
    apply(heuristic=schedule_SPT, data_file="SPT_data",inst=inst,
         batch=batch,
         arr_time=arr_time)

if __name__ == "__main__":
    inst="10050"
    print("Heuristic solution generation for instance:", inst)
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
    main(inst=inst, batch="b_1", arr_time="90")
    print("inst=inst, batch=b_1, arr_time=90: done.")
    main(inst=inst, batch="b_2", arr_time="90")
    print("inst=inst, batch=b_2, arr_time=90: done.")
    main(inst=inst, batch="b_3", arr_time="90")
    print("inst=inst, batch=b_3, arr_time=90: done.")