from Parameters import configs
import numpy as np


def permissibleLeftShift(hour,n_jobs,a, durMat, mchMat, mchsStartTimes, opIDsOnMchs):
    jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(hour,n_jobs,a, mchMat, durMat, mchsStartTimes, opIDsOnMchs)
    dur_a = np.take(durMat, a)
    mch_a = np.take(mchMat, a) - 1
    startTimesForMchOfa = mchsStartTimes[int(mch_a)]
    opsIDsForMchOfa = opIDsOnMchs[int(mch_a)]
    #print("opIDsOnMchs[int(mch_a)]:", opIDsOnMchs[int(mch_a)])
    flag = False

    possiblePos = np.where(jobRdyTime_a < startTimesForMchOfa)[0]
    # print('possiblePos:', possiblePos)
    if len(possiblePos) == 0:
        startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa)
    else:
        idxLegalPos, legalPos, endTimesForPossiblePos = calLegalPos(dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa)
        # print('legalPos:', legalPos)
        if len(legalPos) == 0:
            startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa)
        else:
            flag = True
            startTime_a = putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa)
    #print("startTime_a, flag:", startTime_a, flag)
    return startTime_a, flag


def putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa):
    # index = first position of -config.high in startTimesForMchOfa
    # print('Yes!OK!')
    index = np.where(startTimesForMchOfa == -configs.high)[0][0]
    startTime_a = max(jobRdyTime_a, mchRdyTime_a)
    startTimesForMchOfa[index] = startTime_a
    opsIDsForMchOfa[index] = a
    return startTime_a


def calLegalPos(dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa):
    startTimesOfPossiblePos = startTimesForMchOfa[possiblePos]
    durOfPossiblePos = np.take(durMat, opsIDsForMchOfa[possiblePos])
    startTimeEarlst = max(jobRdyTime_a, startTimesForMchOfa[possiblePos[0]-1] + np.take(durMat, [opsIDsForMchOfa[possiblePos[0]-1]]))
    endTimesForPossiblePos = np.append(startTimeEarlst, (startTimesOfPossiblePos + durOfPossiblePos))[:-1]# end time for last ops don't care
    possibleGaps = startTimesOfPossiblePos - endTimesForPossiblePos
    idxLegalPos = np.where(dur_a <= possibleGaps)[0]
    legalPos = np.take(possiblePos, idxLegalPos)
    return idxLegalPos, legalPos, endTimesForPossiblePos


def putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa):
    earlstIdx = idxLegalPos[0]
    # print('idxLegalPos:', idxLegalPos)
    earlstPos = legalPos[0]
    startTime_a = endTimesForPossiblePos[earlstIdx]
    # print('endTimesForPossiblePos:', endTimesForPossiblePos)
    startTimesForMchOfa[:] = np.insert(startTimesForMchOfa, earlstPos, startTime_a)[:-1]
    opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, a)[:-1]
    return startTime_a


def calJobAndMchRdyTimeOfa(hour,n_jobs,a, mchMat, durMat, mchsStartTimes, opIDsOnMchs):
    #print(a)
    mch_a = np.take(mchMat, a) - 1
    #print("mch_a:", mch_a)
    # cal jobRdyTime_a
    jobPredecessor = a - 1 if a % mchMat.shape[1] != 0 else None
    #print("jobPredecessor:", jobPredecessor)
    if jobPredecessor is not None:
        durJobPredecessor = np.take(durMat, jobPredecessor)
        mchJobPredecessor = np.take(mchMat, jobPredecessor) - 1
        # print(mchJobPredecessor )
        # print("opids:",opIDsOnMchs )
        # print("mchJobPredecessor:",mchJobPredecessor ,mchsStartTimes[int(mchJobPredecessor)], opIDsOnMchs[(mchJobPredecessor)])#[np.where(opIDsOnMchs[int(mchJobPredecessor)] == jobPredecessor)])
        # #print("opIDsOnMchs[int(mchJobPredecessor)]",(mchsStartTimes[int(mchJobPredecessor)][np.where(opIDsOnMchs[int(mchJobPredecessor)] == jobPredecessor)] + durJobPredecessor))
        jobRdyTime_a = (mchsStartTimes[int(mchJobPredecessor)][np.where(opIDsOnMchs[int(mchJobPredecessor)] == jobPredecessor)] + durJobPredecessor).item()
        # print("jobRdyTime_a:", jobRdyTime_a)
    else:
        # j_order=(a//n_jobs) +1
        # if j_order<=n_jobs:
        #     jobRdyTime_a =hour
        # else:
        #     sum= n_jobs
        #     h=1
        #     while sum < j_order:
        #         sum+=added[h]
        #         h+=1
        #     hour_add=h-1
        #     jobRdyTime_a = hour_add
        jobRdyTime_a =hour
    
    #print("jobRdyTime_a:", jobRdyTime_a)
    # cal mchRdyTime_a
    # print("opIDsOnMchs", opIDsOnMchs)
    # print("opIDsOnMchs[(mch_a)]:", opIDsOnMchs[int(mch_a)])
    # print("[np.where(opIDsOnMchs[mch_a] >= 0)]:", (np.where(opIDsOnMchs[int(mch_a)] >= 0)))
    # print("opIDsOnMchs[(mch_a)][np.where(opIDsOnMchs[mch_a] >= 0)]:", opIDsOnMchs[int(mch_a)][np.where(opIDsOnMchs[int(mch_a)] >= 0)])
    
    mchPredecessor = opIDsOnMchs[int(mch_a)][np.where(opIDsOnMchs[int(mch_a)] >= 0)][-1] if len(np.where(opIDsOnMchs[int(mch_a)] >= 0)[0]) != 0 else None
   #print("mchPredecessor: ", mchPredecessor)
    if mchPredecessor is not None:                                                        #@problrm in 
        durMchPredecessor = np.take(durMat, mchPredecessor)
        #print("type is :", type((mchsStartTimes[int(mch_a)][np.where(mchsStartTimes[int(mch_a)] >= 0)][-1] + durMchPredecessor)))
        mchRdyTime_a = (mchsStartTimes[int(mch_a)][np.where(mchsStartTimes[int(mch_a)] >= 0)][-1] + durMchPredecessor).item()
    else:
        mchRdyTime_a = 0
    #print("mchRdyTime_a: ", mchRdyTime_a)
    return jobRdyTime_a, mchRdyTime_a


if __name__ == "__main__":
    from reset_set_0 import SJSSP
    from Inst_generator import uni_instance_gen
    import time

    n_j = 3
    n_m = 3
    low = 1
    high = 99
    SEED = 10
    np.random.seed(SEED)
    env = SJSSP(n_j=n_j, n_m=n_m)

    '''arr = np.ones(3)
    idces = np.where(arr == -1)
    print(len(idces[0]))'''

    # rollout env random action
    t1 = time.time()
    data = uni_instance_gen(n_j=n_j, n_m=n_m, low=low, high=high)
    print('Dur')
    print(data[0])
    print('Mach')
    print(data[-1])
    print()

    # start time of operations on machines
    mchsStartTimes = -configs.high * np.ones_like(data[0].transpose(), dtype=np.int32)
    # Ops ID on machines
    opIDsOnMchs = -n_j * np.ones_like(data[0].transpose(), dtype=np.int32)

    # random rollout to test
    # count = 0
    _, _, omega, mask = env.reset(data)
    rewards = []
    flags = []
    # ts = []
    while True:
        action = np.random.choice(omega[np.where(mask == 0)])
        print(action)
        mch_a = np.take(data[-1], action) - 1
        # print(mch_a)
        # print('action:', action)
        # t3 = time.time()
        adj, _, reward, done, omega, mask = env.step(action)
        # t4 = time.time()
        # ts.append(t4 - t3)
        # jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(a=action, mchMat=data[-1], durMat=data[0], mchsStartTimes=mchsStartTimes, opIDsOnMchs=opIDsOnMchs)
        # print('mchRdyTime_a:', mchRdyTime_a)
        startTime_a, flag = permissibleLeftShift(a=action, durMat=data[0].astype(np.single), mchMat=data[-1], mchsStartTimes=mchsStartTimes, opIDsOnMchs=opIDsOnMchs)
        flags.append(flag)
        # print('startTime_a:', startTime_a)
        # print('mchsStartTimes\n', mchsStartTimes)
        # print('NOOOOOOOOOOOOO' if not np.array_equal(env.mchsStartTimes, mchsStartTimes) else '\n')
        print('opIDsOnMchs\n', opIDsOnMchs)
        # print('LBs\n', env.LBs)
        rewards.append(reward)
        # print('ET after action:\n', env.LBs)
        print()
        if env.done():
            break
    t2 = time.time()
    print(t2 - t1)
    # print(sum(ts))
    # print(np.sum(opIDsOnMchs // n_m, axis=1))
    # print(np.where(mchsStartTimes == mchsStartTimes.max()))
    # print(opIDsOnMchs[np.where(mchsStartTimes == mchsStartTimes.max())])
    #print(mchsStartTimes.max() + np.take(data[0], opIDsOnMchs[np.where(mchsStartTimes == mchsStartTimes.max())]))
    # np.save('sol', opIDsOnMchs // n_m)
    # np.save('jobSequence', opIDsOnMchs)
    # np.save('testData', data)
    # print(mchsStartTimes)
    durAlongMchs = np.take(data[0], opIDsOnMchs)
    mchsEndTimes = mchsStartTimes + durAlongMchs
    print(mchsStartTimes)
    print(mchsEndTimes)
    print()
    print(env.opIDsOnMchs)
    print(env.adj)
    # print(sum(flags))
    # data = np.load('data.npy')

    # print(len(np.where(np.array(rewards) == 0)[0]))
    # print(rewards)
