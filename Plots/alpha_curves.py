
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
arr_time="90"
# List of your Excel files for y1
x=np.array([0,0.25,0.50,0.75,1])
inst="1010"
batch="b_1"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"

files1= [file_1, file_2, file_3, file_4, file_5]

batch="b_2"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"
files2= [file_1, file_2, file_3, file_4, file_5]

batch="b_3"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"
files3= [file_1, file_2, file_3, file_4, file_5]

data_y1 = [pd.read_excel(file)['Objective_function'].values[0] for file in files1]
print("data y1:", data_y1)
data_y1 = [round(float(data),2) for data in data_y1]
data_y2 = [pd.read_excel(file)['Objective_function'].values[0] for file in files2]
data_y2 = [round(float(data),2) for data in data_y2]
data_y3 = [pd.read_excel(file)['Objective_function'].values[0] for file in files3]
data_y3 = [round(float(data),2) for data in data_y3]

# Convert the list into a numpy array
y1 = np.array(data_y1)
y2 = np.array(data_y2)  
y3 = np.array(data_y3)


# Inst15x15
inst="1515"
batch="b_1"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"

files1= [file_1, file_2, file_3, file_4, file_5]

batch="b_2"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"
files2= [file_1, file_2, file_3, file_4, file_5]

batch="b_3"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"
files3= [file_1, file_2, file_3, file_4, file_5]

data_y1 = [pd.read_excel(file)['Objective_function'].values[0] for file in files1]
print("data y1:", data_y1)
data_y1 = [round(float(data),2) for data in data_y1]
data_y2 = [pd.read_excel(file)['Objective_function'].values[0] for file in files2]
data_y2 = [round(float(data),2) for data in data_y2]
data_y3 = [pd.read_excel(file)['Objective_function'].values[0] for file in files3]
data_y3 = [round(float(data),2) for data in data_y3]
a1=np.array(data_y1)
a2=np.array(data_y2)
a3=np.array(data_y3)

# Inst20x15
inst="2015"
batch="b_1"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"

files1= [file_1, file_2, file_3, file_4, file_5]

batch="b_2"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"
files2= [file_1, file_2, file_3, file_4, file_5]

batch="b_3"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"
files3= [file_1, file_2, file_3, file_4, file_5]

data_y1 = [pd.read_excel(file)['Objective_function'].values[0] for file in files1]
print("data y1:", data_y1)
data_y1 = [round(float(data),2) for data in data_y1]
data_y2 = [pd.read_excel(file)['Objective_function'].values[0] for file in files2]
data_y2 = [round(float(data),2) for data in data_y2]
data_y3 = [pd.read_excel(file)['Objective_function'].values[0] for file in files3]
data_y3 = [round(float(data),2) for data in data_y3]
b1=np.array(data_y1)
b2=np.array(data_y2)
b3=np.array(data_y3)


# Inst20x20
inst="2020"
batch="b_1"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"

files1= [file_1, file_2, file_3, file_4, file_5]

batch="b_2"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"
files2= [file_1, file_2, file_3, file_4, file_5]

batch="b_3"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"
files3= [file_1, file_2, file_3, file_4, file_5]

data_y1 = [pd.read_excel(file)['Objective_function'].values[0] for file in files1]
print("data y1:", data_y1)
data_y1 = [round(float(data),2) for data in data_y1]
data_y2 = [pd.read_excel(file)['Objective_function'].values[0] for file in files2]
print("data y2:", data_y2)
data_y2 = [round(float(data),2) for data in data_y2]
data_y3 = [pd.read_excel(file)['Objective_function'].values[0] for file in files3]
print("data y3:", data_y3)
data_y3 = [round(float(data),2) for data in data_y3]


c1=np.array(data_y1)
c2=np.array(data_y2)
c3=np.array(data_y3)

# Inst30x15
inst="3015"
batch="b_1"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"

files1= [file_1, file_2, file_3, file_4, file_5]

batch="b_2"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"
files2= [file_1, file_2, file_3, file_4, file_5]

batch="b_3"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"
files3= [file_1, file_2, file_3, file_4, file_5]

data_y1 = [pd.read_excel(file)['Objective_function'].values[0] for file in files1]
print("data y1:", data_y1)
data_y1 = [round(float(data),2) for data in data_y1]
data_y2 = [pd.read_excel(file)['Objective_function'].values[0] for file in files2]
data_y2 = [round(float(data),2) for data in data_y2]
data_y3 = [pd.read_excel(file)['Objective_function'].values[0] for file in files3]
data_y3 = [round(float(data),2) for data in data_y3]
d1=np.array(data_y1)
d2=np.array(data_y2)
d3=np.array(data_y3)

# Inst30x20
inst="3020"
batch="b_1"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"

files1= [file_1, file_2, file_3, file_4, file_5]

batch="b_2"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"
files2= [file_1, file_2, file_3, file_4, file_5]

batch="b_3"
location1="./"+inst+"/0/"+batch+"/"
location2="./"+inst+"/0.25/"+batch+"/"
location3="./"+inst+"/0.5/"+batch+"/"
location4="./"+inst+"/0.75/"+batch+"/"
location5="./"+inst+"/1/"+batch+"/"
file_1 = location1+f"{inst}_0_{batch}.xlsx"
file_2 = location2+f"{inst}_0.25_{batch}.xlsx"
file_3 = location3+f"{inst}_0.5_{batch}.xlsx"
file_4 = location4+f"{inst}_0.75_{batch}.xlsx"
file_5 = location5+f"{inst}_1_{batch}.xlsx"
files3= [file_1, file_2, file_3, file_4, file_5]

data_y1 = [pd.read_excel(file)['Objective_function'].values[0] for file in files1]
print("data y1:", data_y1)
data_y1 = [round(float(data),2) for data in data_y1]
data_y2 = [pd.read_excel(file)['Objective_function'].values[0] for file in files2]
data_y2 = [round(float(data),2) for data in data_y2]
data_y3 = [pd.read_excel(file)['Objective_function'].values[0] for file in files3]
data_y3 = [round(float(data),2) for data in data_y3]

e1=np.array(data_y1)
e2=np.array(data_y2)
e3=np.array(data_y3)


# Create a figure and a grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
# axes[0, 0].plot(x, y1)
# axes[0, 1].plot(x, y2)

# # Set the limits of y-axis for first plot
# axes[0, 0].set_ylim([0, 20000])  # replace ymin1 and ymax1 with the limits you want for first plot

# # Set the limits of y-axis for second plot
# axes[0, 1].set_ylim([0, 20000])  # replace ymin2 and ymax2 with the limits you want for second plot

# If you want to invert the y-axis for the first plot
# axes[0, 0].invert_yaxis()

# Define colors, labels, and line styles for each plot within each subplot
plot_info = [
    {'color': 'green', 'label': 'Batch = 1','ls':'-'},
    {'color': 'blue', 'label': 'Batch = 2','ls':'-'},
    {'color': 'red', 'label': 'Batch = 3','ls':'-'},
   

]



# Loop through each subplot and add plots with labels and legend
for i, ax in enumerate(axes.flat):
    # print(i)
    for j, info in enumerate(plot_info):
        # print(j)
        if i==0:
            ax.plot(x, [y1, y2, y3][j], label=info['label'],marker="^", color=info['color'], linestyle=info['ls'] )
        if i==1:
            ax.plot(x, [a1, a2, a3][j], label=info['label'],marker="^", color=info['color'], linestyle=info['ls'])
        if i==2:
            ax.plot(x, [b1, b2, b3][j], label=info['label'], marker="^",color=info['color'], linestyle=info['ls'])
        if i==3:
            ax.plot(x, [c1, c2, c3][j], label=info['label'], marker="^",color=info['color'], linestyle=info['ls'])
        if i==4:
            ax.plot(x, [d1, d2, d3][j], label=info['label'], marker="^",color=info['color'], linestyle=info['ls'])
        if i==5:
            ax.plot(x, [e1, e2, e3][j], label=info['label'],marker="^", color=info['color'], linestyle=info['ls'])
    if i==0:
        ax.set_title(f'Inst. 10x10')
    if i==1:
        ax.set_title(f'Inst. 15x15') 
    if i==2:
        ax.set_title(f'Inst. 20x15')    
    if i==3:
        ax.set_title(f'Inst. 20x20')  
    if i==4:
        ax.set_title(f'Inst. 30x15')    
    if i==5:
        ax.set_title(f'Inst. 30x20')


    ax.set_xlabel('ALpha')  # Add x-axis label
    ax.set_ylabel('SUM')  # Add y-axis label
    ax.set_xticks(x)
    ax.legend(loc='upper right')  # Add legend to each subplot
    ax.grid()  # Add grid to each subplot
# Add a title to the entire figure
# fig.suptitle('Subplots with Multiple Plots, Colors, and Labels')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

plt.savefig('./{}{}'.format('Mchat','jpeg'))
