## Overview
This repository contains implementations of Proximal Policy Optimization (PPO) and Constrained PPO (CPPO) algorithms for addressing the real-time Job Shop Scheduling Problem (JSSP) under stochastic job arrivals. 
The code represents research work conducted during my PhD thesis, focusing on applying reinforcement learning techniques to optimize scheduling decisions in dynamic job shop environments.

## Research Context
The Job Shop Scheduling Problem is a classical NP-hard combinatorial optimization problem where a set of jobs, each consisting of a sequence of operations, must be scheduled on a set of machines to optimize certain objectives (e.g., makespan, tardiness).
This research explores how reinforcement learning approaches can improve traditional scheduling algorithms by adapting to unexpected job arrivals in job shop environments.

## Algorithms Implemented
- **PPO-AC (Proximal Policy Optimization Actor & Critic)**: A policy gradient method that uses a clipped surrogate objective function to improve training stability when learning scheduling policies.
- **Constrained PPO-AC**: An extension of PPO that incorporates constraints to ensure scheduling decisions meet specific requirements in the job shop context.

## Key Features
- Implementation of PPO and CPPO algorithms tailored for job shop scheduling problems
- Simulation environment for testing scheduling policies in various job shop configurations
- Performance evaluation metrics (makespan, schedule stability)
- Comparative analysis with traditional job shop scheduling heuristics and algorithms:
          *Priority dispatching rules (e.g. FIFO, SPT)
          *CPOPTIMIZER: IBM Constraint Programming solver
          *IBM MIP solver
## Requirements
Pleaase refer to requirements.txt file

## Running training experiments
For PPO-AC:
```bash
python PPO-AC/Training/PPO_train.py
```
For CPPO-AC:
```bash
python Constrained PPO-AC/Training/CPPO_train.py
```
## Running test experiments
For PPO-AC:
```bash
python PPO-AC/Test/PPO_test.py
```
For CPPO-AC:
```bash
python Constrained PPO-AC/Test/PPO_test.py
```
## Citation
If you use this code in your research, please cite:
```
Nour El Houda Hammami. Vers un ordonnancement en temps réel optimal : apprentissage par renforcement basé sur des politiques pour des systèmes stables et efficaces. Autre [cs.OH].
Université de Bretagne occidentale - Brest; École nationale d'ingénieurs de Tunis (Tunisie), 2024. Français. ⟨NNT : 2024BRES0095⟩. ⟨tel-05025470⟩
```
