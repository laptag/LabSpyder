# -*- coding: utf-8 -*-
#
# CH1 is I_GEA*100k*10*2  - 100k resistor, gain=10 amplifier with 50 ohm output into 1Mohm scope input
# CH2 is VRF(Whistler) / 100   --> IRF = VCH4*100/50,   seriously Aliased
# CH3 is I_beam at 0.1A/V, but should have been 50 ohms and was not, so not sure what beam current
# CH4 is V_beam / 1000
#

if False:
	#doing movies, not interactive:
	import matplotlib
	matplotlib.use("Agg")
	doing_movie = True
#import matplotlib.pyplot as plt
import matplotlib.animation as manimation

import numpy as np
import pylab as plt
import scipy.ndimage as synd

#============================================================================
def smooth(data, interval):
	return synd.filters.uniform_filter(data, interval)
#============================================================================
def subsample(data, interval):
	return np.average(data.reshape(-1,interval), 1)  # slightly odd construction here: treats the 1D array as a collection of columns, then averages over each
#============================================================================
def avg_min_max(data, fract=0.02):
	sorted = np.sort(data)
	npts = np.size(sorted)
	ilo = int(fract*npts)
	ihi = int((1-fract)*npts)
	lo = np.average(sorted[:ilo])
	hi = np.average(sorted[ihi:npts])
	return lo,hi
#============================================================================
# for curve_fit
def fline(x, A, B):
    return A*x + B
#============================================================================

# I need to close the graph after every plt.show(), at least the way I am using matplotlib.
# Also, I want the stupid gray border to be gone
# ...so use this routine instead of plt.show()

def plt_resize_text():
	ax = plt.subplot()
	for ticklabel in (ax.get_xticklabels() + ax.get_yticklabels()):
	    ticklabel.set_fontsize(18)   # this odd construction seems to be necessary to adjust the size of each tic label
	ax.xaxis.get_label().set_fontsize(18)
	ax.yaxis.get_label().set_fontsize(18)
	ax.title.set_fontsize(28)

def plt_show():
	#plt_resize_text()
	plt.rc("figure",facecolor="white")       # get rid of persistent gray border containing axis labels
	plt.show()                               # put it on screen, for interaction
	plt.close()                              # discard traces etc.

#============================================================================

ifn = "Saves/save.npy"

print("Reading file", ifn)

(sIprobe,time,xlocs,ylocs) = np.load(ifn)  # writes file <ofn>.npy

NTimes = np.size(time)
NX     = np.size(xlocs)
NY     = np.size(ylocs)
dt = time[1]-time[0]   # ms

# do a contour plot of the results
frst_ndx = int(0.5/dt)
last_ndx = int(1.25/dt)

for n in range(frst_ndx,last_ndx,10):
	t = time[n]
	plt.title("16-03-05")
	if n == frst_ndx:
		p = plt.imshow(sIprobe[:,:,n])  # first time through: generate plot
		plt.clim(0,.3)              # set color limits
		plt.colorbar()
	else:
		p.set_data(sIprobe[:,:,n])      # subsequently: change the data in the existing plot
	print(n, ": t = ", t)
	plt.pause(0.03)                   # shows latest version

plt_show()
print("ok")
