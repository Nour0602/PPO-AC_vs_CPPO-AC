# Real-time job-shop rescheduling with PPO and constrained PPO

Code accompanying the PhD thesis of **Nour El Houda Hammami** (Université de Bretagne Occidentale & ENIT, 2024):
*Towards optimal real-time scheduling: policy-based reinforcement learning for stable and efficient systems*,
[HAL tel-05025470](https://theses.hal.science/tel-05025470).

This repository implements:
- **PPO-AC**: a Proximal Policy Optimization actor-critic agent, using a graph neural network on the disjunctive
  graph, that reschedules a job shop each time new jobs arrive (objectives: makespan and schedule stability);
- **CPPO-AC**: a constrained variant in which stability is enforced through an adaptive Lagrangian multiplier.

## Related publications
- N. E. H. Hammami, B. Lardeux, A. B. Hadj-Alouane, M. Jridi. *Design and calibration of a DRL algorithm for
  solving the job shop scheduling problem under unexpected job arrivals.* Flexible Services and Manufacturing
  Journal, 37(1):125–156, 2025. https://doi.org/10.1007/s10696-024-09540-2
- N. E. H. Hammami, B. Lardeux, A. B. Hadj-Alouane, M. Jridi. *Job shop scheduling: a novel DRL approach for
  continuous schedule-generation facing real-time job arrivals.* IFAC-PapersOnLine, 55(10):2493–2498, 2022.
  https://doi.org/10.1016/j.ifacol.2022.10.083

## Research Context
The Job Shop Scheduling Problem is a classical NP-hard combinatorial optimization problem where a set of jobs, each consisting of a sequence of operations, must be scheduled on a set of machines to optimize certain objectives (e.g., makespan, tardiness).
This research explores how reinforcement learning approaches can improve traditional scheduling algorithms by adapting to unexpected job arrivals in job shop environments.

## Key Features
- Implementation of PPO and CPPO algorithms tailored for job shop scheduling problems
- Simulation environment for testing scheduling policies in various job shop configurations
- Performance evaluation metrics (makespan, schedule stability)
- Comparative analysis with traditional job shop scheduling methods:
  - priority dispatching rules (e.g. FIFO, SPT);
  - IBM CP Optimizer (constraint programming);
  - IBM CPLEX (mixed-integer programming).
## Requirements
Please refer to requirements.txt file

## Running training experiments
For PPO-AC:
```bash
python PPO-AC/Training/PPO_train.py
```
For CPPO-AC:
```bash
python "Constrained PPO-AC/Training/CPPO_train.py"
```
## Running test experiments
For PPO-AC:
```bash
python "PPO-AC/Test/PPO_test.py"
```
For CPPO-AC:
```bash
python "Constrained PPO-AC/Test/CPPO_test.py"
```
## Citation
If you use this code in your research, please cite:
```
Nour El Houda Hammami. Vers un ordonnancement en temps réel optimal : apprentissage par renforcement basé sur des politiques pour des systèmes stables et efficaces. Autre [cs.OH].
Université de Bretagne occidentale - Brest; École nationale d'ingénieurs de Tunis (Tunisie), 2024. Français. ⟨NNT : 2024BRES0095⟩. ⟨tel-05025470⟩
```
