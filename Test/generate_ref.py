from mb_agg_re import *
from agent_utils_re import *
import torch
import argparse
from Params import configs
import time
import numpy as np
import random

def test():  
    device = torch.device(configs.device)

    parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
    parser.add_argument('--Pn_j', type=int, default=10, help='Number of jobs of instances to test')
    parser.add_argument('--Pn_m', type=int, default=10, help='Number of machines instances to test')
    parser.add_argument('--Nn_j', type=int, default=10, help='Number of jobs on which to be loaded net are trained')
    parser.add_argument('--Nn_m', type=int, default=10, help='Number of machines on which to be loaded net are trained')
    parser.add_argument('--low', type=int, default=1, help='LB of duration')
    parser.add_argument('--high', type=int, default=99, help='UB of duration')
    parser.add_argument('--seed', type=int, default=200, help='Seed for validate set generation')
    params = parser.parse_args()
    
    N_JOBS_P = params.Pn_j
    N_MACHINES_P = params.Pn_m
    LOW = 1
    HIGH = 99
    SEED = params.seed
    N_JOBS_N = params.Nn_j
    N_MACHINES_N = params.Nn_m
    
    
    from JSSP_Env import SJSSP
    from PPO_jssp_multiInstances import PPO
    env = SJSSP(n_j=N_JOBS_P, n_m=N_MACHINES_P)
    #print(env.number_of_jobs)
    
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=N_JOBS_P,
              n_m=N_MACHINES_P,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)
    path = './{}.pth'.format(str(N_JOBS_N) + '_' + str(N_MACHINES_N) + '_' + str(1) + '_' + str(99))
    # path = './{}.pth'.format(str(30) + '_' + str(20) + '_' + str(1) + '_' + str(99))
    # ppo.policy.load_state_dict(torch.load(path))
    ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cuda:0')))
    # ppo.policy.eval()
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                              batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                              n_nodes=env.number_of_tasks,
                              device=device)
    # 34 41 41 57 40 56 63 35 67 66 45 67 51 68 68 41 67 30 65 64
    from uniform_instance_gen_re import uni_instance_gen
    # np.random.seed(SEED)
    
    #result = []
    #torch.cuda.synchronize()
    # LOW=random.randint(1,90)
    # HIGH=random.randint(120,200)
    adj, fea, candidate, mask = env.reset(uni_instance_gen(n_j=N_JOBS_P, n_m=N_MACHINES_P, low=1, high=99))
    #print(env.dur, env.m)
    #print(fea.shape)
    ep_reward = - env.max_endTime
        # delta_t = []
        # t5 = time.time()
    while True:
            # t3 = time.time()
        fea_tensor = torch.from_numpy(fea).to(device)
        adj_tensor = torch.from_numpy(adj).to(device)
        candidate_tensor = torch.from_numpy(candidate).to(device)
        mask_tensor = torch.from_numpy(mask).to(device)
            # t4 = time.time()
            # delta_t.append(t4 - t3)

        with torch.no_grad():
            pi, _ = ppo.policy(x=fea_tensor,
                                graph_pool=g_pool_step,
                                   padded_nei=None,
                                   adj=adj_tensor,
                                   candidate=candidate_tensor.unsqueeze(0),
                                   mask=mask_tensor.unsqueeze(0))
                # action = sample_select_action(pi, omega)
            action = greedy_select_action(pi, candidate)

        adj, fea, reward, done, candidate, mask = env.step(action)
        ep_reward += reward

        if done:
            break

    
    return env.dur, env.m,env.opIDsOnMchs,env.mchsStartTimes,env.temp1, env.partial_sol_sequeence

#print(test())


