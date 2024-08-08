
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

x=np.array([1,2,3])
arr_time="90"
inst="4030"

inst=inst

location1="./"+inst+"/LG/"+"b_1"+"/"+arr_time+"/"
location2="./"+inst+"/LG/"+"b_2"+"/"+arr_time+"/"
location3="./"+inst+"/LG/"+"b_3"+"/"+arr_time+"/"
file_1 = location1+f"{inst}_b_1.xlsx"
file_2 = location2+f"{inst}_b_2.xlsx"
file_3 = location3+f"{inst}_b_3.xlsx"
files= [file_1, file_2, file_3]

data_y1 = [pd.read_excel(file)['Mean_ms'].values[0] for file in files]
print("data y1:", data_y1)
data_y1 = [round(float(data),2) for data in data_y1]
# data_y2 = [pd.read_excel(file)['Mean_ms'].values[3] for file in files]
# data_y2 = [round(float(data),2) for data in data_y2]
# data_y3 = [pd.read_excel(file)['Mean_ms'].values[4] for file in files]
# data_y3 = [round(float(data),2) for data in data_y3]
data_y4 = [pd.read_excel(file)['Mean_ms'].values[1] for file in files]
data_y4 = [round(float(data),2) for data in data_y4]
data_y5 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
data_y5 = [round(float(data),2) for data in data_y5]
data_y6 = [pd.read_excel(file)['Mean_ms'].values[3] for file in files]
data_y6 = [round(float(data),2) for data in data_y6]
# data_y7 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
# # data_y7 = [-20000, -20000, -20000]
# data_y7 = [round(float(data),2) for data in data_y7]
data_y8 = [pd.read_excel(file)['Initial_MS'].values[2] for file in files]
data_y8 = [round(float(data),2) for data in data_y8]

y1=np.array(data_y1)
y2=np.array(data_y4)
y3=np.array(data_y5)
y4=np.array(data_y6)
y8=np.array(data_y8)
# a3=np.array(data_y3)
# a4=np.array(data_y4)
# a5=np.array(data_y5)
# a6=np.array(data_y6)
# a7=np.array(data_y7)
# Inst20x20
inst="5050"
inst=inst
# arr_time="30"
location1="./"+inst+"/LG/"+"b_1"+"/"+arr_time+"/"
location2="./"+inst+"/LG/"+"b_2"+"/"+arr_time+"/"
location3="./"+inst+"/LG/"+"b_3"+"/"+arr_time+"/"
file_1 = location1+f"{inst}_b_1.xlsx"
file_2 = location2+f"{inst}_b_2.xlsx"
file_3 = location3+f"{inst}_b_3.xlsx"
files= [file_1, file_2, file_3]

data_y1 = [pd.read_excel(file)['Mean_ms'].values[0] for file in files]
print("data y1:", data_y1)
data_y1 = [round(float(data),2) for data in data_y1]
# data_y2 = [pd.read_excel(file)['Mean_ms'].values[3] for file in files]
# data_y2 = [round(float(data),2) for data in data_y2]
# data_y3 = [pd.read_excel(file)['Mean_ms'].values[4] for file in files]
# data_y3 = [round(float(data),2) for data in data_y3]
data_y4 = [pd.read_excel(file)['Mean_ms'].values[1] for file in files]
data_y4 = [round(float(data),2) for data in data_y4]
data_y5 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
data_y5 = [round(float(data),2) for data in data_y5]
data_y6 = [pd.read_excel(file)['Mean_ms'].values[3] for file in files]
data_y6 = [round(float(data),2) for data in data_y6]
# data_y7 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
# # data_y7 = [-20000, -20000, -20000]
# data_y7 = [round(float(data),2) for data in data_y7]
data_y8 = [pd.read_excel(file)['Initial_MS'].values[2] for file in files]
data_y8 = [round(float(data),2) for data in data_y8]

a1=np.array(data_y1)
a2=np.array(data_y4)
a3=np.array(data_y5)
a4=np.array(data_y6)
a8=np.array(data_y8)
# b3=np.array(data_y3)
# b4=np.array(data_y4)
# b5=np.array(data_y5)
# b6=np.array(data_y6)
# b7=np.array(data_y7)
# Inst 20x015
inst="7050"
inst=inst
# arr_time="30"
location1="./"+inst+"/LG/"+"b_1"+"/"+arr_time+"/"
location2="./"+inst+"/LG/"+"b_2"+"/"+arr_time+"/"
location3="./"+inst+"/LG/"+"b_3"+"/"+arr_time+"/"
file_1 = location1+f"{inst}_b_1.xlsx"

file_2 = location2+f"{inst}_b_2.xlsx"
file_3 = location3+f"{inst}_b_3.xlsx"
files= [file_1, file_2, file_3]

data_y1 = [pd.read_excel(file)['Mean_ms'].values[0] for file in files]
print("data y1:", data_y1)
data_y1 = [round(float(data),2) for data in data_y1]
# data_y2 = [pd.read_excel(file)['Mean_ms'].values[3] for file in files]
# data_y2 = [round(float(data),2) for data in data_y2]
# data_y3 = [pd.read_excel(file)['Mean_ms'].values[4] for file in files]
# data_y3 = [round(float(data),2) for data in data_y3]
data_y4 = [pd.read_excel(file)['Mean_ms'].values[1] for file in files]
data_y4 = [round(float(data),2) for data in data_y4]
data_y5 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
data_y5 = [round(float(data),2) for data in data_y5]
data_y6 = [pd.read_excel(file)['Mean_ms'].values[3] for file in files]
data_y6 = [round(float(data),2) for data in data_y6]
# data_y7 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
# # data_y7 = [-20000, -20000, -20000]
# data_y7 = [round(float(data),2) for data in data_y7]
data_y8 = [pd.read_excel(file)['Initial_MS'].values[2] for file in files]
data_y8 = [round(float(data),2) for data in data_y8]


b1=np.array(data_y1)
b2=np.array(data_y4)
b3=np.array(data_y5)
b4=np.array(data_y6)
b8=np.array(data_y8)
# b3=np.array(data_y3)
# b4=np.array(data_y4)
# b5=np.array(data_y5)
# b6=np.array(data_y6)
# b7=np.array(data_y7)
# Inst20x20
inst="9050"
inst=inst
# arr_time="30"
location1="./"+inst+"/LG/"+"b_1"+"/"+arr_time+"/"
location2="./"+inst+"/LG/"+"b_2"+"/"+arr_time+"/"
location3="./"+inst+"/LG/"+"b_3"+"/"+arr_time+"/"
file_1 = location1+f"{inst}_b_1.xlsx"
file_2 = location2+f"{inst}_b_2.xlsx"
file_3 = location3+f"{inst}_b_3.xlsx"
files= [file_1, file_2, file_3]

data_y1 = [pd.read_excel(file)['Mean_ms'].values[0] for file in files]
print("data y1 9050:", data_y1)
data_y1 = [round(float(data),2) for data in data_y1]
# data_y2 = [pd.read_excel(file)['Mean_ms'].values[3] for file in files]
# data_y2 = [round(float(data),2) for data in data_y2]
# data_y3 = [pd.read_excel(file)['Mean_ms'].values[4] for file in files]
# data_y3 = [round(float(data),2) for data in data_y3]
data_y4 = [pd.read_excel(file)['Mean_ms'].values[1] for file in files]
data_y4 = [round(float(data),2) for data in data_y4]
data_y5 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
data_y5 = [round(float(data),2) for data in data_y5]
data_y6 = [pd.read_excel(file)['Mean_ms'].values[3] for file in files]
data_y6 = [round(float(data),2) for data in data_y6]
# data_y7 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
# # data_y7 = [-20000, -20000, -20000]
# data_y7 = [round(float(data),2) for data in data_y7]
data_y8 = [pd.read_excel(file)['Initial_MS'].values[2] for file in files]
data_y8 = [round(float(data),2) for data in data_y8]

c1=np.array(data_y1)
c2=np.array(data_y4)
c3=np.array(data_y5)
c4=np.array(data_y6)
c8=np.array(data_y8)
# c3=np.array(data_y3)
# c4=np.array(data_y4)
# c5=np.array(data_y5)
# c6=np.array(data_y6)
# c7=np.array(data_y7)
# c7=np.array([0,0,0])
# Inst30x15
inst="10050"
inst=inst
# arr_time="30"
location1="./"+inst+"/LG/"+"b_1"+"/"+arr_time+"/"
location2="./"+inst+"/LG/"+"b_2"+"/"+arr_time+"/"
location3="./"+inst+"/LG/"+"b_3"+"/"+arr_time+"/"
file_1 = location1+f"{inst}_b_1.xlsx"
file_2 = location2+f"{inst}_b_2.xlsx"
file_3 = location3+f"{inst}_b_3.xlsx"
files= [file_1, file_2, file_3]

data_y1 = [pd.read_excel(file)['Mean_ms'].values[0] for file in files]
print("data y1:", data_y1)
data_y1 = [round(float(data),2) for data in data_y1]
# data_y2 = [pd.read_excel(file)['Mean_ms'].values[3] for file in files]
# data_y2 = [round(float(data),2) for data in data_y2]
# data_y3 = [pd.read_excel(file)['Mean_ms'].values[4] for file in files]
# data_y3 = [round(float(data),2) for data in data_y3]
data_y4 = [pd.read_excel(file)['Mean_ms'].values[1] for file in files]
data_y4 = [round(float(data),2) for data in data_y4]
data_y5 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
data_y5 = [round(float(data),2) for data in data_y5]
data_y6 = [pd.read_excel(file)['Mean_ms'].values[3] for file in files]
data_y6 = [round(float(data),2) for data in data_y6]
# data_y7 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
# # data_y7 = [-20000, -20000, -20000]
# data_y7 = [round(float(data),2) for data in data_y7]
data_y8 = [pd.read_excel(file)['Initial_MS'].values[2] for file in files]
data_y8 = [round(float(data),2) for data in data_y8]



d1=np.array(data_y1)
d2=np.array(data_y4)
d3=np.array(data_y5)
d4=np.array(data_y6)
d8=np.array(data_y8)
# d3=np.array(data_y3)
# d4=np.array(data_y4)
# d5=np.array(data_y5)
# d6=np.array(data_y6)
# d7=np.array(data_y7)



# # Inst30x20
# inst="3020"
# inst=inst
# # arr_time="30"
# location1="./"+inst+"/LG/"+"b_1"+"/"+arr_time+"/"
# location2="./"+inst+"/LG/"+"b_2"+"/"+arr_time+"/"
# location3="./"+inst+"/LG/"+"b_3"+"/"+arr_time+"/"
# file_1 = location1+f"{inst}_b_1.xlsx"
# file_2 = location2+f"{inst}_b_2.xlsx"
# file_3 = location3+f"{inst}_b_3.xlsx"
# files= [file_1, file_2, file_3]

# data_y1 = [pd.read_excel(file)['Mean_ms'].values[0] for file in files]
# print("data y1:", data_y1)
# data_y1 = [round(float(data),2) for data in data_y1]
# # data_y2 = [pd.read_excel(file)['Mean_ms'].values[3] for file in files]
# # data_y2 = [round(float(data),2) for data in data_y2]
# # data_y3 = [pd.read_excel(file)['Mean_ms'].values[4] for file in files]
# # data_y3 = [round(float(data),2) for data in data_y3]
# data_y4 = [pd.read_excel(file)['Mean_ms'].values[1] for file in files]
# data_y4 = [round(float(data),2) for data in data_y4]
# data_y5 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
# data_y5 = [round(float(data),2) for data in data_y5]
# data_y6 = [pd.read_excel(file)['Mean_ms'].values[3] for file in files]
# data_y6 = [round(float(data),2) for data in data_y6]
# # data_y7 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
# # # data_y7 = [-20000, -20000, -20000]
# # data_y7 = [round(float(data),2) for data in data_y7]
# data_y8 = [pd.read_excel(file)['Initial_MS'].values[2] for file in files]
# data_y8 = [round(float(data),2) for data in data_y8]

# e1=np.array(data_y1)
# e2=np.array(data_y4)
# e3=np.array(data_y5)
# e4=np.array(data_y6)
# e8=np.array(data_y8)
# # e3=np.array(data_y3)
# # e4=np.array(data_y4)
# # e5=np.array(data_y5)
# # e6=np.array(data_y6)
# # e7=np.array(data_y7)




# Create a figure and a grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))



plot_info = [
    {'color': 'green', 'label': 'CPPO-AC','ls':'-'},
    {'color': 'blue', 'label': 'CPOPT_1','ls':'-'},
    {'color': 'orange', 'label': 'CPOPT_0','ls':'-'},
    {'color': 'red', 'label': 'CPOPT_75','ls':'-'},
    {'color': 'maroon', 'label': 'MS_Init','ls':'--'},

]

for i, ax in enumerate(axes.flat):
    # print(i)
    for j, info in enumerate(plot_info):
        # print(j)
        if i==0:
            ax.plot(x, [y1, y2,y3,y4,y8][j], label=info['label'],linestyle=info['ls'],marker="^", color=info['color'] )
        if i==1:
            ax.plot(x, [a1, a2,a3,a4,a8][j], label=info['label'],linestyle=info['ls'],marker="^", color=info['color'])
        if i==2:
            ax.plot(x, [b1, b2,b3,b4,b8][j], label=info['label'],linestyle=info['ls'], marker="^",color=info['color'])
        if i==3:
            ax.plot(x, [c1, c2,c3,c4,c8][j], label=info['label'],linestyle=info['ls'], marker="^",color=info['color'])
        if i==4:
            ax.plot(x, [d1, d2,d3,d4,d8][j], label=info['label'],linestyle=info['ls'],marker="^",color=info['color'])
        # if i==5:
        #     ax.plot(x, [e1, e2,e3,e4,e8][j], label=info['label'],linestyle=info['ls'],marker="^", color=info['color'])
        if i==5:
            axes[-1, -1].axis('off') 
    if i==0:
        ax.set_title(f'Inst. 40x30')
    if i==1:
        ax.set_title(f'Inst. 50x50') 
    if i==2:
        ax.set_title(f'Inst. 70x50')    
    if i==3:
        ax.set_title(f'Inst. 90x50')  
    if i==4:
        ax.set_title(f'Inst. 100x50')    
    # if i==5:
    #     ax.set_title(f'Inst. 30x20')
    if i==5:
            axes[-1, -1].axis('off') 

    ax.set_xlabel('Batch')  # Add x-axis label
    ax.set_ylabel('MS')  # Add y-axis label
    ax.set_xticks(x)
    ax.legend(loc='upper right')  # Add legend to each subplot
    ax.grid()
    
# Add a title to the entire figure
# fig.suptitle('Subplots with Multiple Plots, Colors, and Labels')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()

plt.savefig('./{}{}'.format('Mchat','jpeg'))
