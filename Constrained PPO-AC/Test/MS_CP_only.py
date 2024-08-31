
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
# eXAMPLE OF PLOTTING INST 15X15
# arr_time="30"
# # List of your Excel files for y1
x=np.array([1,2,3])
# # inst="1010"
# inst="1515"
# batch="b_1"
# # arr_time="30"
# location1="./"+inst+"/NOLG/"+"b_1"+"/"+arr_time+"/"
# location2="./"+inst+"/NOLG/"+"b_2"+"/"+arr_time+"/"
# location3="./"+inst+"/NOLG/"+"b_3"+"/"+arr_time+"/"
# file_1 = location1+f"{inst}_b_1.xlsx"
# file_2 = location2+f"{inst}_b_2.xlsx"
# file_3 = location3+f"{inst}_b_3.xlsx"

# # print("file 1:", file_1)
# # files_y1 = ['file1_y1.xlsx', 'file2_y1.xlsx', 'file3_y1.xlsx']  # replace with your file names
# files= [file_1, file_2, file_3]

# # Read the first 'DE' value from each file into a list
# data_y1 = [pd.read_excel(file)['Mean_ms'].values[0] for file in files]
# print("data y1:", data_y1)
# data_y1 = [round(float(data),2) for data in data_y1]
# data_y2 = [pd.read_excel(file)['Mean_ms'].values[3] for file in files]
# data_y2 = [round(float(data),2) for data in data_y2]
# data_y3 = [pd.read_excel(file)['Mean_ms'].values[4] for file in files]
# data_y3 = [round(float(data),2) for data in data_y3]
# data_y4 = [pd.read_excel(file)['Mean_ms'].values[5] for file in files]
# data_y4 = [round(float(data),2) for data in data_y4]
# data_y5 = [pd.read_excel(file)['Mean_ms'].values[6] for file in files]
# data_y5 = [round(float(data),2) for data in data_y5]
# data_y6 = [pd.read_excel(file)['Mean_ms'].values[1] for file in files]
# data_y6 = [round(float(data),2) for data in data_y6]
# data_y7 = [pd.read_excel(file)['Mean_ms'].values[2] for file in [file_1, file_2]]
# data_y7.append(-20000)
# data_y7 = [round(float(data),2) for data in data_y7]

# # Convert the list into a numpy array
# y1 = np.array(data_y1)
# y2 = np.array(data_y2)  
# y3 = np.array(data_y3)
# y4 = np.array(data_y4)
# y5 = np.array(data_y5)
# y6 = np.array(data_y6)
# y7 = np.array(data_y7)


# Inst15x15
arr_time="30"
# inst="1515"
# inst=inst
# # arr_time="30"
# location1="./"+inst+"/NOLG/"+"b_1"+"/"+arr_time+"/"
# location2="./"+inst+"/NOLG/"+"b_2"+"/"+arr_time+"/"
# location3="./"+inst+"/NOLG/"+"b_3"+"/"+arr_time+"/"
# file_1 = location1+f"{inst}_b_1.xlsx"
# file_2 = location2+f"{inst}_b_2.xlsx"
# file_3 = location3+f"{inst}_b_3.xlsx"
# files= [file_1, file_2, file_3]

# data_y1 = [pd.read_excel(file)['Mean_ms'].values[0] for file in files]
# print("data y1:", data_y1)
# data_y1 = [round(float(data),2) for data in data_y1]
# data_y2 = [pd.read_excel(file)['Mean_ms'].values[3] for file in files]
# data_y2 = [round(float(data),2) for data in data_y2]
# data_y3 = [pd.read_excel(file)['Mean_ms'].values[4] for file in files]
# data_y3 = [round(float(data),2) for data in data_y3]
# data_y4 = [pd.read_excel(file)['Mean_ms'].values[5] for file in files]
# data_y4 = [round(float(data),2) for data in data_y4]
# data_y5 = [pd.read_excel(file)['Mean_ms'].values[6] for file in files]
# data_y5 = [round(float(data),2) for data in data_y5]
# data_y6 = [pd.read_excel(file)['Mean_ms'].values[1] for file in files]
# data_y6 = [round(float(data),2) for data in data_y6]
# data_y7 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
# # data_y7.append(-20000)
# print("data y7:", data_y7)
# data_y7 = [round(float(data),2) for data in data_y7]

# a1=np.array(data_y1)
# a2=np.array(data_y2)
# a3=np.array(data_y3)
# a4=np.array(data_y4)
# a5=np.array(data_y5)
# a6=np.array(data_y6)
# a7=np.array(data_y7)

# # Inst20x15
# # inst="2015"
# arr_time="90"
# inst="1515"
# # arr_time="30"
# location1="./"+inst+"/NOLG/"+"b_1"+"/"+arr_time+"/"
# location2="./"+inst+"/NOLG/"+"b_2"+"/"+arr_time+"/"
# location3="./"+inst+"/NOLG/"+"b_3"+"/"+arr_time+"/"
# file_1 = location1+f"{inst}_b_1.xlsx"
# file_2 = location2+f"{inst}_b_2.xlsx"
# file_3 = location3+f"{inst}_b_3.xlsx"
# files= [file_1, file_2, file_3]

# data_y1 = [pd.read_excel(file)['Mean_ms'].values[0] for file in files]
# print("data y1:", data_y1)
# data_y1 = [round(float(data),2) for data in data_y1]
# data_y2 = [pd.read_excel(file)['Mean_ms'].values[3] for file in files]
# data_y2 = [round(float(data),2) for data in data_y2]
# data_y3 = [pd.read_excel(file)['Mean_ms'].values[4] for file in files]
# data_y3 = [round(float(data),2) for data in data_y3]
# data_y4 = [pd.read_excel(file)['Mean_ms'].values[5] for file in files]
# data_y4 = [round(float(data),2) for data in data_y4]
# data_y5 = [pd.read_excel(file)['Mean_ms'].values[6] for file in files]
# data_y5 = [round(float(data),2) for data in data_y5]
# data_y6 = [pd.read_excel(file)['Mean_ms'].values[1] for file in files]
# data_y6 = [round(float(data),2) for data in data_y6]
# data_y7 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
# # data_y7 = [-20000, -20000, -20000]
# data_y7 = [round(float(data),2) for data in data_y7]

# b1=np.array(data_y1)
# b2=np.array(data_y2)
# b3=np.array(data_y3)
# b4=np.array(data_y4)
# b5=np.array(data_y5)
# b6=np.array(data_y6)
# b7=np.array(data_y7)
# # b7=np.array([0,0,0])

# # Create a figure and a grid of subplots
# fig, axes = plt.subplots(1, 3, figsize=(16, 5))


# # Define colors, labels, and line styles for each plot within each subplot
# plot_info = [
#     {'color': 'green', 'label': 'PPO-AC',  'ls':'-'},
#     {'color': 'yellow', 'label': 'LPT','ls':'-'},
#     {'color': 'purple', 'label': 'SPT','ls':'-'},
#     {'color': 'orange', 'label': 'FIFO','ls':'-'},
#     {'color': 'cyan', 'label': 'LIFO','ls':'-'},
#     {'color': 'blue', 'label': 'CPOPT','ls':'-'},
#     {'color': 'red', 'label': 'MIP','ls':'-'}

# ]

# # plot_info = [
# #     {'color': 'green', 'label': 'PPO-AC'},
# #     {'color': 'blue', 'label': 'CPOPT'},
# #     {'color': 'red', 'label': 'MIP'}

# # ]

# # Loop through each subplot and add plots with labels and legend
# for i, ax in enumerate(axes.flat):
#     # print(i)
#     for j, info in enumerate(plot_info):
#         # print(j)
#         if i==0:
#             ax.plot(x, [y1, y2, y3,y4,y5,y6,y7][j], label=info['label'],marker="^", color=info['color'] )
#         if i==1:
#             ax.plot(x, [a1, a2, a3,a4,a5,a6,a7][j], label=info['label'],marker="^", color=info['color'])
#         if i==2:
#             ax.plot(x, [b1, b2, b3,b4,b5,b6,b7][j], label=info['label'], marker="^",color=info['color'])
       
#     if i==0:
#         ax.set_title(f'Inst. 15x15: Arrival time at the begining')
#     if i==1:
#         ax.set_title(f'Inst. 15x15: Arrival time in the middle') 
#     if i==2:
#         ax.set_title(f'Inst. 15x15:: Arrival time at the end')    
   


#     ax.set_xlabel('Batch')  # Add x-axis label
#     ax.set_ylabel('DEV')  # Add y-axis label
#     ax.set_xticks(x)
#     ax.legend(loc='upper right')  # Add legend to each subplot
#     ax.grid()
    
# # Add a title to the entire figure
# # fig.suptitle('Subplots with Multiple Plots, Colors, and Labels')

# # Adjust spacing between subplots
# plt.tight_layout()

# # Show the plot
# plt.show()

# plt.savefig('./{}{}'.format('Mchat','jpeg'))






# Above the plots for three time for instance 1515
# Inst10X10
# arr_time="30"
# # List of your Excel files for y1
x=np.array([1,2,3])
inst="1010"
arr_time="30"
inst=inst

location1="./"+inst+"/NOLG/"+"b_1"+"/"+arr_time+"/"
location2="./"+inst+"/NOLG/"+"b_2"+"/"+arr_time+"/"
location3="./"+inst+"/NOLG/"+"b_3"+"/"+arr_time+"/"
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
# data_y4 = [pd.read_excel(file)['Mean_ms'].values[5] for file in files]
# data_y4 = [round(float(data),2) for data in data_y4]
# data_y5 = [pd.read_excel(file)['Mean_ms'].values[6] for file in files]
# data_y5 = [round(float(data),2) for data in data_y5]
data_y6 = [pd.read_excel(file)['Mean_ms'].values[1] for file in files]
data_y6 = [round(float(data),2) for data in data_y6]
# data_y7 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
# # data_y7 = [-20000, -20000, -20000]
# data_y7 = [round(float(data),2) for data in data_y7]
data_y8 = [pd.read_excel(file)['Initial_MS'].values[2] for file in files]
data_y8 = [round(float(data),2) for data in data_y8]

y1=np.array(data_y1)
y2=np.array(data_y6)
# a3=np.array(data_y3)
# a4=np.array(data_y4)
# a5=np.array(data_y5)
# a6=np.array(data_y6)
# a7=np.array(data_y7)
y8=np.array(data_y8)
# Inst20x20
inst="1515"
inst=inst
# arr_time="30"
location1="./"+inst+"/NOLG/"+"b_1"+"/"+arr_time+"/"
location2="./"+inst+"/NOLG/"+"b_2"+"/"+arr_time+"/"
location3="./"+inst+"/NOLG/"+"b_3"+"/"+arr_time+"/"
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
# data_y4 = [pd.read_excel(file)['Mean_ms'].values[5] for file in files]
# data_y4 = [round(float(data),2) for data in data_y4]
# data_y5 = [pd.read_excel(file)['Mean_ms'].values[6] for file in files]
# data_y5 = [round(float(data),2) for data in data_y5]
data_y6 = [pd.read_excel(file)['Mean_ms'].values[1] for file in files]
data_y6 = [round(float(data),2) for data in data_y6]
# data_y7 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
# # data_y7 = [-20000, -20000, -20000]
# data_y7 = [round(float(data),2) for data in data_y7]
data_y8 = [pd.read_excel(file)['Initial_MS'].values[2] for file in files]
data_y8 = [round(float(data),2) for data in data_y8]


a1=np.array(data_y1)
a2=np.array(data_y6)
# b3=np.array(data_y3)
# b4=np.array(data_y4)
# b5=np.array(data_y5)
# b6=np.array(data_y6)
# b7=np.array(data_y7)
a8=np.array(data_y8)
# Inst 20x015
inst="2015"
inst=inst
# arr_time="30"
location1="./"+inst+"/NOLG/"+"b_1"+"/"+arr_time+"/"
location2="./"+inst+"/NOLG/"+"b_2"+"/"+arr_time+"/"
location3="./"+inst+"/NOLG/"+"b_3"+"/"+arr_time+"/"
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
# data_y4 = [pd.read_excel(file)['Mean_ms'].values[5] for file in files]
# data_y4 = [round(float(data),2) for data in data_y4]
# data_y5 = [pd.read_excel(file)['Mean_ms'].values[6] for file in files]
# data_y5 = [round(float(data),2) for data in data_y5]
data_y6 = [pd.read_excel(file)['Mean_ms'].values[1] for file in files]
data_y6 = [round(float(data),2) for data in data_y6]
# data_y7 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
# # data_y7 = [-20000, -20000, -20000]
# data_y7 = [round(float(data),2) for data in data_y7]
data_y8 = [pd.read_excel(file)['Initial_MS'].values[2] for file in files]
data_y8 = [round(float(data),2) for data in data_y8]


b1=np.array(data_y1)
b2=np.array(data_y6)
# b3=np.array(data_y3)
# b4=np.array(data_y4)
# b5=np.array(data_y5)
# b6=np.array(data_y6)
# b7=np.array(data_y7)
b8=np.array(data_y8)
# Inst20x20
inst="2020"
inst=inst
# arr_time="30"
location1="./"+inst+"/NOLG/"+"b_1"+"/"+arr_time+"/"
location2="./"+inst+"/NOLG/"+"b_2"+"/"+arr_time+"/"
location3="./"+inst+"/NOLG/"+"b_3"+"/"+arr_time+"/"
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
# data_y4 = [pd.read_excel(file)['Mean_ms'].values[5] for file in files]
# data_y4 = [round(float(data),2) for data in data_y4]
# data_y5 = [pd.read_excel(file)['Mean_ms'].values[6] for file in files]
# data_y5 = [round(float(data),2) for data in data_y5]
data_y6 = [pd.read_excel(file)['Mean_ms'].values[1] for file in files]
print("data y6:", data_y6)
data_y6 = [round(float(data),2) for data in data_y6]
# data_y7 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
# # data_y7 = [-20000, -20000, -20000]
# data_y7 = [round(float(data),2) for data in data_y7]
data_y8 = [pd.read_excel(file)['Initial_MS'].values[2] for file in files]
data_y8 = [round(float(data),2) for data in data_y8]


c1=np.array(data_y1)
c2=np.array(data_y6)
# c3=np.array(data_y3)
# c4=np.array(data_y4)
# c5=np.array(data_y5)
# c6=np.array(data_y6)
# c7=np.array(data_y7)
# c7=np.array([0,0,0])
c8=np.array(data_y8)
# Inst30x15
inst="3015"
inst=inst
# arr_time="30"
location1="./"+inst+"/NOLG/"+"b_1"+"/"+arr_time+"/"
location2="./"+inst+"/NOLG/"+"b_2"+"/"+arr_time+"/"
location3="./"+inst+"/NOLG/"+"b_3"+"/"+arr_time+"/"
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
# data_y4 = [pd.read_excel(file)['Mean_ms'].values[5] for file in files]
# data_y4 = [round(float(data),2) for data in data_y4]
# data_y5 = [pd.read_excel(file)['Mean_ms'].values[6] for file in files]
# data_y5 = [round(float(data),2) for data in data_y5]
data_y6 = [pd.read_excel(file)['Mean_ms'].values[1] for file in files]
data_y6 = [round(float(data),2) for data in data_y6]
# data_y7 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
# # data_y7 = [-20000, -20000, -20000]
# data_y7 = [round(float(data),2) for data in data_y7]
data_y8 = [pd.read_excel(file)['Initial_MS'].values[2] for file in files]
data_y8 = [round(float(data),2) for data in data_y8]

d1=np.array(data_y1)
d2=np.array(data_y6)
# d3=np.array(data_y3)
# d4=np.array(data_y4)
# d5=np.array(data_y5)
# d6=np.array(data_y6)
# d7=np.array(data_y7)
d8=np.array(data_y8)



# Inst30x20
inst="3020"
inst=inst
# arr_time="30"
location1="./"+inst+"/NOLG/"+"b_1"+"/"+arr_time+"/"
location2="./"+inst+"/NOLG/"+"b_2"+"/"+arr_time+"/"
location3="./"+inst+"/NOLG/"+"b_3"+"/"+arr_time+"/"
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
# data_y4 = [pd.read_excel(file)['Mean_ms'].values[5] for file in files]
# data_y4 = [round(float(data),2) for data in data_y4]
# data_y5 = [pd.read_excel(file)['Mean_ms'].values[6] for file in files]
# data_y5 = [round(float(data),2) for data in data_y5]
data_y6 = [pd.read_excel(file)['Mean_ms'].values[1] for file in files]
data_y6 = [round(float(data),2) for data in data_y6]
# data_y7 = [pd.read_excel(file)['Mean_ms'].values[2] for file in files]
# # data_y7 = [-20000, -20000, -20000]
# data_y7 = [round(float(data),2) for data in data_y7]
data_y8 = [pd.read_excel(file)['Initial_MS'].values[2] for file in files]
data_y8 = [round(float(data),2) for data in data_y8]


e1=np.array(data_y1)
e2=np.array(data_y6)
# e3=np.array(data_y3)
# e4=np.array(data_y4)
# e5=np.array(data_y5)
# e6=np.array(data_y6)
# e7=np.array(data_y7)
e8=np.array(data_y8)




# Create a figure and a grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(12, 8))


# Define colors, labels, and line styles for each plot within each subplot
# plot_info = [
#     {'color': 'green', 'label': 'PPO-AC',  'ls':'-'},
#     {'color': 'yellow', 'label': 'LPT','ls':'-'},
#     {'color': 'purple', 'label': 'SPT','ls':'-'},
#     {'color': 'orange', 'label': 'FIFO','ls':'-'},
#     {'color': 'cyan', 'label': 'LIFO','ls':'-'},
#     {'color': 'blue', 'label': 'CPOPT','ls':'-'},
#     {'color': 'red', 'label': 'MIP','ls':'-'}

# ]

plot_info = [
    {'color': 'green', 'label': 'PPO-AC'},
    {'color': 'blue', 'label': 'CPOPT'},
    {'color': 'maroon', 'label': 'MS_Init','ls':'--'}
    

]

# Loop through each subplot and add plots with labels and legend
# for i, ax in enumerate(axes.flat):
#     # print(i)
#     for j, info in enumerate(plot_info):
#         # print(j)
#         if i==0:
#             ax.plot(x, [y1, y2, y3,y4,y5,y6,y7][j], label=info['label'],marker="^", color=info['color'] )
#         if i==1:
#             ax.plot(x, [a1, a2, a3,a4,a5,a6,a7][j], label=info['label'],marker="^", color=info['color'])
#         if i==2:
#             ax.plot(x, [b1, b2, b3,b4,b5,b6,b7][j], label=info['label'], marker="^",color=info['color'])
#         if i==3:
#             ax.plot(x, [c1, c2, c3,c4,c5,c6,c7][j], label=info['label'], marker="^",color=info['color'])
#         if i==4:
#             ax.plot(x, [d1, d2, d3,d4,d5,d6,d7][j], label=info['label'], marker="^",color=info['color'])
#         if i==5:
#             ax.plot(x, [e1, e2, e3,e4,e5,e6,e7][j], label=info['label'],marker="^", color=info['color'])
#     if i==0:
#         ax.set_title(f'Inst. 10x10')
#     if i==1:
#         ax.set_title(f'Inst. 15x15') 
#     if i==2:
#         ax.set_title(f'Inst. 20x15')    
#     if i==3:
#         ax.set_title(f'Inst. 20x20')  
#     if i==4:
#         ax.set_title(f'Inst. 30x15')    
#     if i==5:
#         ax.set_title(f'Inst. 30x20')


for i, ax in enumerate(axes.flat):
    # print(i)
    for j, info in enumerate(plot_info):
        # print(j)
        if i==0:
            ax.plot(x, [y1, y2,y8][j], label=info['label'],marker="^", color=info['color'] )
        if i==1:
            ax.plot(x, [a1, a2,a8][j], label=info['label'],marker="^", color=info['color'])
        if i==2:
            ax.plot(x, [b1, b2,b8][j], label=info['label'], marker="^",color=info['color'])
        if i==3:
            ax.plot(x, [c1, c2,c8][j], label=info['label'], marker="^",color=info['color'])
        if i==4:
            ax.plot(x, [d1, d2,d8][j], label=info['label'], marker="^",color=info['color'])
        if i==5:
            ax.plot(x, [e1, e2,e8][j], label=info['label'],marker="^", color=info['color'])
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
