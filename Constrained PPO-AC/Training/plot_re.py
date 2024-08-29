import numpy
import numpy as np
import matplotlib.pyplot as plt
import re

# plot parameters
x_label_scale = 20
y_label_scale = 20
anchor_text_size = 15
show = True
save = True
save_file_type = '.jpeg'
# problem params
n_j = 15
n_m = 15
l = 1
h = 99
# s=100
stride =1

# datatype ='vloss'
datatype ='mvg_cost'
#'vali1','vali2', 'log', 'vloss', 'loss'
f5= open('moving_rew_15_15_1_99.txt', 'r').readline()
f1 = open('moving_sum_15_15_1_99.txt', 'r').readline()
f = open('valiMSP_15_15_1_99.txt', 'r').readline()
f2= open('valiDEVP_15_15_1_99.txt', 'r').readline()
f3=open('VLoss_15_15_1_99.txt', 'r').readline()
f4=open('moving_cost_15_15_1_99.txt', 'r').readline()
if datatype == 'vali1':
    obj = numpy.array([float(s) for s in re.findall(r'-?\d+\.?\d*', f)[0::1]])[:].reshape(-1, stride).mean(axis=-1)
  
    idx = np.arange(obj.shape[0])
  
    # plotting...
    plt.xlabel('Iteration', {'size': x_label_scale})
    plt.ylabel('MakeSpan ', {'size': y_label_scale})
    plt.grid()
    plt.plot(idx, obj, color='tab:blue', label='{}x{}'.format(n_j, n_m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('ms', save_file_type))
    if show:
        plt.show()

elif datatype == 'vali2':
    obj = numpy.array([float(s) for s in re.findall(r'-?\d+\.?\d*', f2)[0::1]])[:].reshape(-1, stride).mean(axis=-1)
    idx = np.arange(obj.shape[0])
    obj1=obj[:35200]
    # obj1=obj1.reshape(-1, s).mean(axis=-1)
    idx1 =np.arange(352)
    # plotting...
    plt.xlabel('Iteration', {'size': x_label_scale})
    plt.ylabel('DEVIATION ', {'size': y_label_scale})
    plt.grid()
    plt.plot(idx, obj, color='tab:blue', label='{}x{}'.format(n_j, n_m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('de', save_file_type))
    if show:
        plt.show()


elif datatype == 'loss':
    obj = numpy.array([float(s) for s in re.findall(r'-?\d+\.?\d*', f4)[1::2]])[:].reshape(-1, stride).mean(axis=-1)
    idx = np.arange(obj.shape[0])
    # plotting...
    plt.xlabel('Iteration', {'size': x_label_scale})
    plt.ylabel('Loss', {'size': y_label_scale})
    plt.grid()
    plt.plot(idx, obj, color='tab:green', label='{}x{}'.format(n_j, n_m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('message-passing_time', save_file_type))
    if show:
        plt.show()
elif datatype == 'vloss':
    obj = numpy.array([float(s) for s in re.findall(r'-?\d+\.?\d*', f3)[0::1]])[:].reshape(-1, stride).mean(axis=-1)
    idx = np.arange(obj.shape[0])
    # plotting...
    plt.xlabel('Iteration', {'size': x_label_scale})
    plt.ylabel('V-Loss ', {'size': y_label_scale})
    plt.grid()
    plt.plot(idx, obj, color='tab:blue', label='{}x{}'.format(n_j, n_m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('vloss', save_file_type))
    if show:
        plt.show()

elif datatype == 'mvg_cost':
    obj = numpy.array([float(s) for s in re.findall(r'-?\d+\.?\d*', f4)[0::1]])[:].reshape(-1, stride).mean(axis=-1)
    idx = np.arange(obj.shape[0])
    # plotting...
    plt.xlabel('Iteration', {'size': x_label_scale})
    plt.ylabel('Cost ', {'size': y_label_scale})
    plt.grid()
    plt.plot(idx, obj, color='tab:blue', label='{}x{}'.format(n_j, n_m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('mvg_cost', save_file_type))
    if show:
        plt.show()
elif datatype == 'mvg_sum':
    obj = numpy.array([float(s) for s in re.findall(r'-?\d+\.?\d*', f1)[0::1]])[:].reshape(-1, stride).mean(axis=-1)
    idx = np.arange(obj.shape[0])
    # obj1= obj[]
    # idx1 =np.arange(600)
    # plotting...
    plt.xlabel('Iteration', {'size': x_label_scale})
    plt.ylabel('0.25MS+0.75DE', {'size': y_label_scale})
    plt.grid()
    plt.plot(idx, obj, color='tab:green', label='{}x{}'.format(n_j, n_m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('mvg_sum', save_file_type))
    if show:
        plt.show()
elif datatype == 'mvg_rew':
    obj = numpy.array([float(s) for s in re.findall(r'-?\d+\.?\d*', f5)[0::1]])[:].reshape(-1, stride).mean(axis=-1)
    obj1=obj[:101]
    print("shape", obj.shape[0])
    idx = np.arange(obj.shape[0])
    idx1= np.arange(101)
    # plotting...
    plt.xlabel('Iteration', {'size': x_label_scale})
    plt.ylabel('REWARD ', {'size': y_label_scale})
    plt.grid()
    plt.plot(idx, obj, color='tab:blue', label='{}x{}'.format(n_j, n_m))
    plt.tight_layout()
    plt.legend(fontsize=anchor_text_size)
    if save:
        plt.savefig('./{}{}'.format('mvg_rew', save_file_type))
    if show:
        plt.show()        
else:
    print('Wrong datatype.')





