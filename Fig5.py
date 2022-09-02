#!/usr/bin/env python

__version__ = "Time-stamp: <2018-12-06 11:55:22 ycopin>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

"""
Example of use of TaylorDiagram. Illustration dataset courtesy of Michael
Rawlins.

Rawlins, M. A., R. S. Bradley, H. F. Diaz, 2012. Assessment of regional climate
model simulation estimates over the Northeast United States, Journal of
Geophysical Research (2012JGRD..11723112R).
"""

from taylorDiagram_hy2 import TaylorDiagram
import numpy as NP
import matplotlib.pyplot as PLT

# Reference std
stdrefs = dict(winter=0.75)

# Sample std,rho: Be sure to check order and that correct numbers are placed!
# mae, r2, rmse
samples = dict(winter=[[0.55, 0.84, 0.83, "DGRN"],
                       [1.05, 0.48, 1.49, "LR"],
                       [0.64, 0.77, 0.99, "RF"],
                       [0.71, 0.71, 1.11, "SVM"],
                       [0.56, 0.83, 0.85, "GPR"],
                       [0.57, 0.84, 0.83, "ANN"],
                       [0.57, 0.82, 0.87, "MLPNN"],
                       [0.66, 0.79, 0.94, "ResNet"]])

# Colormap (see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps)
colors = PLT.matplotlib.cm.Set1(NP.linspace(0,1,len(samples['winter'])))

# Here set placement of the points marking 95th and 99th significance
# levels. For more than 102 samples (degrees freedom > 100), critical
# correlation levels are 0.195 and 0.254 for 95th and 99th
# significance levels respectively. Set these by eyeball using the
# standard deviation x and y axis.

#x95 = [0.01, 0.68] # For Tair, this is for 95th level (r = 0.195)
#y95 = [0.0, 3.45]
#x99 = [0.01, 0.95] # For Tair, this is for 99th level (r = 0.254)
#y99 = [0.0, 3.45]

x95 = [0.05, 13.9] # For Prcp, this is for 95th level (r = 0.195)
y95 = [0.0, 71.0]
x99 = [0.05, 19.0] # For Prcp, this is for 99th level (r = 0.254)
y99 = [0.0, 70.0]

rects = dict(winter=111)

fig = PLT.figure(figsize=(6,4))
# fig.suptitle("Talyor-diagram of lnKD", size='x-large')

# for season in ['winter','spring','summer','autumn']:
season = 'winter'

dia = TaylorDiagram(stdrefs[season], fig=fig, rect=rects[season],
                label='Reference')

dia.ax.plot(x95,y95,color='k')
dia.ax.plot(x99,y99,color='k')

# Add samples to Taylor diagram
for i,(stddev,corrcoef,rmse,name) in enumerate(samples[season]):
    dia.add_sample(stddev, corrcoef,
                    marker='$%d$' % (i+1), ms=10, ls='',
                    #mfc='k', mec='k', # B&W
                    mfc=colors[i], mec=colors[i], # Colors
                    label=name)

# Add RMS contours, and label them
contours = dia.add_contours(rms=rmse, levels=10, colors='0.5') # 5 levels
dia.ax.clabel(contours, inline=1, fontsize=10, fmt='%.2f')
# Tricky: ax is the polar ax (used for plots), _ax is the
# container (used for layout)
# dia._ax.set_title(season.capitalize())
# dia._ax.set_title("Talyor-diagram of lnKD")

# Add a figure legend and title. For loc option, place x,y tuple inside [ ].
# Can also use special options here:
# http://matplotlib.sourceforge.net/users/legend_guide.html

fig.legend(dia.samplePoints,
           [ p.get_label() for p in dia.samplePoints ],
           numpoints=1, prop=dict(size='small'), loc='upper right')

fig.tight_layout()

PLT.savefig('./SDD_talyor_diagram.png',dpi=1000)
PLT.show()
