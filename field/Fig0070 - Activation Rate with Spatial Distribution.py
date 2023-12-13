from mylib.statistic_test import *

code_id = '0070 - Activation Rate with Spatial Distribution'
loc = os.path.join(figpath, code_id)
mkdir(loc)

idx = np.where((f1['MiceID'] != 11092)&(f1['MiceID'] != 11095)&(f1['maze_type'] != 0))[0]
if os.path.exists(join(figdata, code_id+'.pkl')):
    with open(join(figdata, code_id+'.pkl'), 'rb') as handle:
        Data = pickle.load(handle)
else:
    Data = DataFrameEstablish(variable_names = ['Position', 'Bin', 'Activation Rate', 'Is Perfect'], 
                              f = f1, function = ActivationRateSpatialPosition_Interface, file_idx=idx,
                              file_name = code_id, behavior_paradigm = 'CrossMaze')

idx = np.where((Data['MiceID'] == 10227)&(Data['Stage'] == 'Stage 2')&(Data['Maze Type'] == 'Maze 1'))[0]
SubData = SubDict(Data, Data.keys(), idx)
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
sns.lineplot(
    x='Training Day',
    y='Activation Rate',
    hue='Is Perfect',
    data = SubData,
)
"""
ax.hist(SubData['Activation Rate'][SubData['Is Perfect'] == 1], bins = 50, range=(0,1), alpha = 0.5)
ax.hist(SubData['Activation Rate'][SubData['Is Perfect'] == 0], bins = 50, range=(0,1), alpha = 0.5)

DP1 = DPs[1]
CP1 = correct_paths[1]
fig = plt.figure(figsize=(4,2))
ax = Clear_Axes(plt.axes(), close_spines=['top', 'right'], ifxticks=True, ifyticks=True)
for i in DP1:
    x = np.where(CP1 == i)[0][0]
    ax.fill_betweenx([0, 1], x-0.5, x+0.5, color = 'gray', alpha = 0.5, edgecolor=None)
sns.lineplot(
    x = 'Bin',
    y = 'Activation Rate',
    hue = 'Is Perfect',
    data = SubData,
)
"""
plt.show()

    
    