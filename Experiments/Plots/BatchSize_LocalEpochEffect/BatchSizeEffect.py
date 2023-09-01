
import numpy as np

import matplotlib.pyplot as plt
fs = 16


fs_l=18

plt.rcParams['figure.figsize'] = (22, 18)

import matplotlib
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
fig, axs = plt.subplots(4, 3)


plt.rcParams["font.family"] = "Times New Roman"
batchSizesPlot = ['64','128','256','512','1024']





# LMRA

axs[0, 0].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/LMRA_FP_ours_bz.npy'), 'x-', color='blue', label='Ours')
axs[0, 0].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/LMRA_FP_global_bz.npy'), '+-',color='orange', label='Last-returned')
axs[0, 0].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/LMRA_FP_last_bz.npy'), 's-', color='green',label='Global')
axs[0, 0].set_title("LMRA", fontsize=24, weight='bold')

axs[0, 0].set_ylabel('Similarity to optimal local model', fontsize=fs_l)
axs[0, 0].set_ylim([0.5, 1])
axs[0, 0].legend(fontsize = fs, loc=2, ncol=2)


axs[1, 0].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/LMRA_AD_ours_bz.npy'), 'x-', color='blue', label='Ours')
axs[1, 0].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/LMRA_AD_global_bz.npy'), '+-',color='orange', label='Last-returned')
axs[1, 0].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/LMRA_AD_last_bz.npy'),  's-',color='green',label='Global')
axs[1, 0].set_ylim([0.5, 1])
axs[1, 0].legend(fontsize = fs, loc=2, ncol=2)
axs[1, 0].set_ylabel('Similarity to optimal local model', fontsize=fs_l)


axs[2, 0].plot(batchSizesPlot,  np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/LMRA_Leaf_ours_bz.npy'), 'x-', color='blue', label='Ours')
axs[2, 0].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/LMRA_Leaf_global_bz.npy'), '+-',color='orange', label='Last-returned')
axs[2, 0].plot(batchSizesPlot,  np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/LMRA_Leaf_last_bz.npy'), 's-' ,color='green',label='Global')
axs[2, 0].set_ylim([0.5, 0.9])
axs[2, 0].legend(fontsize = fs, loc=2, ncol=2)
axs[2, 0].set_ylabel('Similarity to optimal local model', fontsize=fs_l)


axs[3, 0].plot(['4','8','16','32','64'],np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/LMRA_AT_ours_bz.npy'), 'x-', color='blue', label='Ours')
axs[3, 0].plot(['4','8','16','32','64'], np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/LMRA_AT_global_bz.npy'), '+-',color='orange', label='Last-returned')
axs[3, 0].plot(['4','8','16','32','64'], np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/LMRA_AT_last_bz.npy'), 's-', color='green',label='Global')
axs[3, 0].set_ylabel('Similarity to optimal local model', fontsize=fs_l)
axs[3, 0].set_ylim([0.4, 0.9])
axs[3, 0].legend(fontsize = fs, loc=2, ncol=2)
axs[3, 0].set_xlabel('Batch size' , fontsize=20)



axs[3,1].plot(0, label='NA')
axs[3,1].legend(fontsize = 30, loc=2, ncol=2)
axs[3,1].set_xticks([])
axs[3,1].set_yticks([])

axs[3, 1].set_xlabel('Batch size' , fontsize=20)

axs[0, 1].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/AIA_FP_ours_bz.npy'), 'x-', color='blue', label='Ours')
axs[0, 1].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/AIA_FP_last_bz.npy'),'+-',color='orange', label='Last-returned')
axs[0, 1].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/AIA_FP_global_bz.npy'), 's-', color='green',label='Global')
axs[0, 1].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/AIA_FP_SOTA_bz.npy'), 'v-', color='violet', label='SOTA')

axs[0, 1].set_title("AIA", fontsize=24, weight='bold')

axs[0, 1].set_ylabel('Attack accuracy', fontsize=fs_l)
axs[0, 1].set_ylim([0.5, 1])
axs[0, 1].legend(fontsize = fs, loc=2, ncol=2)


axs[1, 1].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/AIA_AD_ours_bz.npy'), 'x-', color='blue', label='Ours')
axs[1, 1].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/AIA_AD_last_bz.npy'),'+-',color='orange', label='Last-returned')
axs[1, 1].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/AIA_AD_global_bz.npy'), 's-', color='green',label='Global')
axs[1, 1].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/AIA_AD_SOTA_bz.npy'), 'v-', color='violet', label='SOTA')
axs[1, 1].set_ylabel('Attack accuracy', fontsize=fs_l)
axs[1, 1].set_ylim([0.5, 1])
axs[1, 1].legend(fontsize = fs, loc=2, ncol=2)




axs[2, 1].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/AIA_Leaf_ours_bz.npy'), 'x-', color='blue', label='Ours')
axs[2, 1].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/AIA_Leaf_last_bz.npy'),'+-',color='orange', label='Last-returned')
axs[2, 1].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/AIA_Leaf_global_bz.npy'), 's-', color='green',label='Global')
axs[2, 1].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/AIA_Leaf_SOTA_bz.npy'), 'v-', color='violet', label='SOTA')
axs[2, 1].set_ylabel('Attack accuracy', fontsize=fs_l)
axs[2, 1].set_ylim([0.5, 1])
axs[2, 1].legend(fontsize = fs, loc=2, ncol=2)
axs[2, 1].set_ylabel('Attack accuracy', fontsize=fs_l)







axs[0, 2].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/SIA_FP_ours_bz.npy'), 'x-', color='blue',label='Ours')
axs[0, 2].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/SIA_FP_last_bz.npy'), '*-', color='orange',label='Last-returned')
axs[0, 2].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/SIA_FP_SOTA_bz.npy'), 'v-', color='violet',label='SOTA')
axs[0, 2].set_title("SIA", fontsize=24, weight='bold')
axs[3, 2].set_xlabel('Batch size' , fontsize=20)

axs[0, 2].set_ylabel('Attack accuracy', fontsize=fs_l)
axs[0, 2].set_ylim([0.4, 1])
axs[0, 2].legend(fontsize = fs, loc=2, ncol=2)


axs[1, 2].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/SIA_AD_ours_bz.npy'), 'x-', color='blue',label='Ours')
axs[1, 2].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/SIA_AD_last_bz.npy'), '*-', color='orange',label='Last-returned')
axs[1, 2].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/SIA_AD_SOTA_bz.npy'), 'v-', color='violet',label='SOTA')
axs[1, 2].set_ylabel('Attack accuracy', fontsize=fs_l)
axs[1, 2].set_ylim([0.4, 1])
axs[1, 2].legend(fontsize = fs, loc=2, ncol=2)


axs[2, 2].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/SIA_Leaf_ours_bz.npy'), 'x-', color='blue',label='Ours')
axs[2, 2].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/SIA_Leaf_last_bz.npy'), '*-', color='orange',label='Last-returned')
axs[2, 2].plot(batchSizesPlot, np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/SIA_Leaf_SOTA_bz.npy'), 'v-', color='violet',label='SOTA')
axs[2, 2].set_ylabel('Attack accuracy', fontsize=fs_l)
axs[2, 2].set_ylim([0.4, 1])
axs[2, 2].legend(fontsize = fs, loc=2, ncol=2)

axs[3, 2].plot(['4','8','16','32','64'], np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/SIA_AT_ours_bz.npy'), 'x-', color='blue',label='Ours')
axs[3, 2].plot(['4','8','16','32','64'], np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/SIA_AT_last_bz.npy'), '*-', color='orange',label='Last-returned')
axs[3, 2].plot(['4','8','16','32','64'], np.load('/Users/idriouich/Documents/PhDworkSpace/PETs/Experiments/SIA_AT_SOTA_bz.npy'), 'v-', color='violet',label='SOTA')
axs[3, 2].set_ylabel('Attack accuracy', fontsize=fs_l)
axs[3, 2].set_ylim([0.4, 0.8])
axs[3, 2].legend(fontsize = fs, loc=2, ncol=2)




fig.delaxes(axs[3,1])
plt.gcf().text(0.4, 0.2, "------------------------------------------",  weight='bold', family= "Times New Roman" ,fontsize=25)



plt.gcf().text(-0.004, 0.2, "AT&T",  weight='bold', family= "Times New Roman" ,fontsize=23)

plt.gcf().text(-0.004, 0.4, "Leaf Synthetic ",  weight='bold', family= "Times New Roman" ,fontsize=23)
plt.gcf().text(-0.004, 0.6, "Adult",  weight='bold', family= "Times New Roman" ,fontsize=23)
plt.gcf().text(-0.004, 0.8, "Flight Prices",  weight='bold', family= "Times New Roman" ,fontsize=23)


#plt.savefig('/Users/idriouich/Desktop/BatchSize.png', dpi=400)

