# Running in local workstation
# High-resolution SDD mapping
# Write by Yuan He
# 2021-08

from numpy.lib.utils import byte_bounds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

plt.rc('font',family='Times New Roman')

df = pd.read_csv('./data/landsat8.csv')
var_name = ['B1','B2','B3','B4','B5','B7']

# Band infor
optical_data = df[var_name].values/10000/math.pi
optical_mean = np.nanmean(optical_data, 0)

# SDD value
sdd_data = df['SDD'].values

print('Average in-situ Rrs:' + str(optical_mean))
plt.figure(figsize=(7,6))
plt.text(-0.1, 0.09, '(a)', fontsize=40,weight='bold')
for i in range(optical_data.shape[0]):
	plt.plot(optical_data[i,:], 'grey', alpha=0.2)
del i
plt.plot(optical_mean, 'ro--')
plt.grid()
plt.xticks([0,1,2,3,4,5],
	['B1','B2','B3','B4','B5','B7'], fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Rrs (SR$^{-1}$)',fontsize=28)
plt.xlabel('Bands', fontsize=28)
plt.tight_layout()
plt.savefig('./Fig2a.png', dpi=1000, bbox_inches='tight')
# plt.show()

print('Min in-situ value:' + str(np.nanmin(sdd_data)))
print('Max in-situ value:' + str(np.nanmax(sdd_data)))
print('Average in-situ value:' + str(np.nanmean(sdd_data)))
plt.figure(figsize=(7,6))
plt.text(12, 0.45, '(b)', fontsize=40,weight='bold')
plt.hist(sdd_data, bins=14, alpha=.4, density=True, rwidth=0.95,
	lw=1, ec="yellow", fc="green")
plt.xlim([0,13])
plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],
	[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],fontsize=20)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5],[0, 0.1, 0.2, 0.3, 0.4, 0.5],fontsize=20)
plt.ylabel('Probability',fontsize=28)
plt.xlabel('In-situ SDD (m)', fontsize=28)
plt.tight_layout()
plt.savefig('./Fig2b.png', dpi=1000, bbox_inches='tight')
# plt.show()