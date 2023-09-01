# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 17:58:17 2022

@author: idriouich
"""

import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib

plt.rcParams["font.family"] = "Times New Roman"
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
plt.rcParams["font.family"] = "Times New Roman"
a_heur = np.load('utils/FlightModelBasedAIA.npy')
b_heur = np.load('utils/FlightSOTAAIA.npy')



dat1 = pd.DataFrame(np.array([a_heur[0,:],b_heur[0,:]]).T, columns=['Ours_local','LBFGS_opt']).assign(dataSize=10)
dat2 = pd.DataFrame(np.array([a_heur[1,:],b_heur[1,:]]).T, columns=['Ours_local','LBFGS_opt']).assign(dataSize=20)
dat3 = pd.DataFrame(np.array([a_heur[2,:],b_heur[2,:]]).T, columns=['Ours_local','LBFGS_opt']).assign(dataSize=50)
dat4 = pd.DataFrame(np.array([a_heur[3,:],b_heur[3,:]]).T, columns=['Ours_local','LBFGS_opt']).assign(dataSize=100)
dat5 = pd.DataFrame(np.array([a_heur[4,:],b_heur[4,:]]).T, columns=['Ours_local','LBFGS_opt']).assign(dataSize=1000)


cdf = pd.concat([dat1, dat2, dat3, dat4, dat5])    
mdf = pd.melt(cdf, id_vars=['dataSize'], var_name=['Attribute inference attack - Neural network regression'])
print(mdf.head())



#    Location Letter     value
# 0         1      A  0.223565
# 1         1      A  0.515797
# 2         1      A  0.377588
# 3         1      A  0.687614
# 4         1      A  0.094116

ax = sns.boxplot(x="dataSize", y="value", hue="Attribute inference attack - Neural network regression", data=mdf, showmeans=True, palette=['blue','pink'])    
ax.set(xlabel='Data size', ylabel='AIA accuracy')
ax.set_xlabel('Data size', fontsize=18)
ax.set_ylabel('AIA accuracy', fontsize=18)
one = mpatches.Patch(facecolor ='blue', label='Ours', linewidth = 0.5, edgecolor = 'black')
two = mpatches.Patch(facecolor='pink', label = 'SOTA', linewidth = 0.5, edgecolor = 'black')


legend = plt.legend(handles=[one, two], loc = 3, fontsize = 16, fancybox = True)
ax.set_title('Flight Prices',fontsize=18)
plt.ylim(0.4, 1)
#plt.savefig('FlightBox.png', dpi=800,  bbox_inches = "tight")

plt.show()


###### Adult           

a_heur = np.load('utils/AdultModelBasedAIA.npy')
b_heur = np.load('utils/AdultSOTAAIA.npy')

dat1 = pd.DataFrame(np.array([a_heur[0,:],b_heur[0,:]]).T, columns=['Ours_local','LBFGS_opt']).assign(dataSize=10)
dat2 = pd.DataFrame(np.array([a_heur[1,:],b_heur[1,:]]).T, columns=['Ours_local','LBFGS_opt']).assign(dataSize=20)
dat3 = pd.DataFrame(np.array([a_heur[2,:],b_heur[2,:]]).T, columns=['Ours_local','LBFGS_opt']).assign(dataSize=50)
dat4 = pd.DataFrame(np.array([a_heur[3,:],b_heur[3,:]]).T, columns=['Ours_local','LBFGS_opt']).assign(dataSize=75)
dat5 = pd.DataFrame(np.array([a_heur[4,:],b_heur[4,:]]).T, columns=['Ours_local','LBFGS_opt']).assign(dataSize=100)


cdf = pd.concat([dat1, dat2, dat3, dat4, dat5])    
mdf = pd.melt(cdf, id_vars=['dataSize'], var_name=['Attribute inference attack - Neural network classification'])
print(mdf.head())



#    Location Letter     value
# 0         1      A  0.223565
# 1         1      A  0.515797
# 2         1      A  0.377588
# 3         1      A  0.687614
# 4         1      A  0.094116

ax = sns.boxplot(x="dataSize", y="value", hue="Attribute inference attack - Neural network classification", data=mdf, showmeans=True,palette=['blue','pink'])    
ax.set(xlabel='Data size', ylabel='AIA accuracy')
ax.set_xlabel('Data size', fontsize=18)
ax.set_ylabel('AIA accuracy', fontsize=18)

one = mpatches.Patch(facecolor ='blue', label='Ours', linewidth = 0.5, edgecolor = 'black')
two = mpatches.Patch(facecolor='pink', label = 'SOTA', linewidth = 0.5, edgecolor = 'black')

legend = plt.legend(handles=[one, two], loc = 3, fontsize = 16, fancybox = True)
ax.set_title('Adult',fontsize=18)
plt.ylim(0.4, 1)
#plt.savefig('AdultBox.png', dpi=800,  bbox_inches = "tight")

plt.show()




##### Synthetic


a_heur = np.load('utils/SyntheticModelBasedAIA.npy')
b_heur = np.load('utils/SyntheticSOTAAIA.npy')

dat1 = pd.DataFrame(np.array([a_heur[0,:],b_heur[0,:]]).T, columns=['Ours_local','LBFGS_opt']).assign(dataSize=10)
dat2 = pd.DataFrame(np.array([a_heur[1,:],b_heur[1,:]]).T, columns=['Ours_local','LBFGS_opt']).assign(dataSize=20)
dat3 = pd.DataFrame(np.array([a_heur[2,:],b_heur[2,:]]).T, columns=['Ours_local','LBFGS_opt']).assign(dataSize=50)
dat4 = pd.DataFrame(np.array([a_heur[3,:],b_heur[3,:]]).T, columns=['Ours_local','LBFGS_opt']).assign(dataSize=100)
dat5 = pd.DataFrame(np.array([a_heur[4,:],b_heur[4,:]]).T, columns=['Ours_local','LBFGS_opt']).assign(dataSize=1000)


cdf = pd.concat([dat1, dat2, dat3, dat4, dat5])    
mdf = pd.melt(cdf, id_vars=['dataSize'], var_name=['Attribute inference attack - Neural network classification'])
print(mdf.head())



#    Location Letter     value
# 0         1      A  0.223565
# 1         1      A  0.515797
# 2         1      A  0.377588
# 3         1      A  0.687614
# 4         1      A  0.094116

ax = sns.boxplot(x="dataSize", y="value", hue="Attribute inference attack - Neural network classification", data=mdf, showmeans=True,palette=['blue','pink'])    
ax.set(xlabel='Data size', ylabel='AIA accuracy')
ax.set_xlabel('Data size', fontsize=18)
ax.set_ylabel('AIA accuracy', fontsize=18)
legend = plt.legend(handles=[one, two], loc = 3, fontsize = 16, fancybox = True)
ax.set_title('Leaf Synthetic',fontsize=18)
plt.ylim(0.4, 1)
#plt.savefig('LeafBox.png', dpi=800,  bbox_inches = "tight")

plt.show()



