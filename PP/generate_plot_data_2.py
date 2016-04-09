# -*- coding: utf-8 -*-
#
# CH1 is I_GEA*100k*10*2  - 100k resistor, gain=10 amplifier with 50 ohm output into 1Mohm scope input
# CH2 is VRF(Whistler) / 100   --> IRF = VCH4*100/50,   seriously Aliased
# CH3 is I_beam at 0.1A/V, but should have been 50 ohms and was not, so not sure what beam current
# CH4 is V_beam / 1000
#

import numpy as np
import pylab as plt
import scipy.ndimage as synd
#from scipy import stats        # for linregress
#from scipy.optimize import curve_fit
from gsmooth import gsmooth
import h5py  as h5py  # HDF5 support

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

# Acquisition particulars:
delta_x = 1.         # unit    TODO: what are these?
delta_y = 1.         # unit          mm?
Resistor = 1e6         # ohms

NX     = 41    # adjusted later
NY     = 41    # adjusted later
NTimes = 10000 # adjusted later
time_per_div = 200e-6

#============================

# get rid of persistent gray border containing axis labels
plt.rc("figure",facecolor="white")

# Read in data from hdf5 file
print("reading data")

fileName = r"P:\Data\Laptag\16-03-05 eBeam & GEA\3_7new_16WhistlerBeamVelocityAnalyzer_B40G_P6e-4T.hdf5"

f = h5py.File(fileName,  "r")

# List the names of the file data objects:
if False:
    print ("objects in hdf5 file are:")
    list_of_names = []
    f.visit(list_of_names.append)
    for i in range(len(list_of_names)):
        print( "   " + list_of_names[i])

# data
Iprobe = f["Acquisition/WS424_1/Channel1"].value
zth    = f["Control/NI7340_ZTheta_1/NI7340_ZTheta_Setup_Arrray"].value

f.close()
#============================


# locations, # entries in data series
NLocs,NTimes = np.shape(Iprobe)

NShots = int(NLocs/np.size(zth))
if np.size(zth)*NShots != NLocs:
    print("OOPS Internal consistency check failed: NLocs = ", NLocs/NShots, ",  but zth only has ",np.size(zth)," entries")

# Count how many Z locations have the same initial Theta location:
for NX in range(0,NLocs):
    if zth[NX][2] != zth[0][2]:     # zth = ...(3, -36.819, -81.837)(4, -32.558, -81.837) ...
        break;

NLocs = int(NLocs/NShots)
NY = int(NLocs / NX)

print("# spatial locations = ", NLocs, "     # data series points = ", NTimes)
if NY*NX != NLocs:
	print("OOPS CHECK NY & NX:  NY*NX = ", NY*NX, ", BUT THE HDF5 FILE HAS ", NLocs, " VALUES")
else:
	print("    ", NLocs," = ", NX, " axial locations X ", NY, " theta locations")
print("# shots at each location = ",NShots)

Iprobe = np.reshape(Iprobe, [NY, NX, NTimes])
Iprobe /= -Resistor;

digrate  = (NTimes/10)/time_per_div
dt       = 1./digrate # sec
dt *= 1000   #ms

# compute time, space vectors
time  = np.arange(NTimes) * dt;                  # like findgen() to create the appropriate vector (in microseconds)
xlocs = np.arange(NX) * delta_x
ylocs = np.arange(NY) * delta_y

nt_beg = 50
nt_end = NTimes - 50
#============================

lines = np.zeros_like(Iprobe)

if False:
	nfit = int(0.3/dt);
	print("fitting first",nfit,"points")

	for j in range(NY):
		# print("j =",j)
		for i in range(NX):
			#1:    lin_coef = np.polyfit(time[0:nfit-1], Iprobe[i,j,0:nfit-1], 1)    then     lines[i,j,:] = lin_coef[0]*time + lin_coef[1]
			#2:    slope, intercept, r_value, p_value, std_err = stats.linregress(time[0:nfit-1], Iprobe[i,j,0:nfit-1])       then      slope*time[k] + intercept
			#3:    A,B = curve_fit(fline, time[0:nfit-1], Iprobe[i,j,0:nfit-1])[0]   then    	lines[i,j,:] = A*time + B
			# very slow:	for k in range(NTimes):
			#				Iprobe[i,j,k] -= A*time[k] + B
			lin_coef = np.polyfit(time[0:nfit-1], Iprobe[i,j,0:nfit-1], 1)
			lines[i,j,:] = lin_coef[0]*time + lin_coef[1]
			#Iprobe[i,j,:] -= lines[i,j,:]

#============================

if True:
	t1 = int( 0.28 / dt)
	t2 = int( 0.33 / dt)
	t3 = int( 1.70 / dt)
	t4 = int( 1.75 / dt)
	t12   = np.average(time[t1:t2])
	t34   = np.average(time[t3:t4])
	for j in range(NY):
		for i in range(NX):
			avg12 = np.average(Iprobe[i,j,t1:t2])
			avg34 = np.average(Iprobe[i,j,t3:t4])
			slope = (avg34-avg12) / (t34-t12)
			intercept = avg12 - slope*t12
			lines[i,j,:] = slope*time + intercept
			#Iprobe[i,j,:] -= lines[i,j,:]



#============================

if True:
	print("smoothing")
	sIprobe = np.zeros_like(Iprobe)

	for j in range(NY):
		for i in range(NX):
			sIprobe[i,j,:] = gsmooth(Iprobe[i,j,:], 50) * 1e6
else:
	sIprobe = Iprobe * 1e6

lines *= 1e6

#============================
# subtract baselines

for j in range(NY):
	for i in range(NX):
		sIprobe[i,j,:] -= lines[i,j,:]

#============================

if True:
	nax = 4         # generate array of nax X nax plots for various spatial locations
	fig, axarr = plt.subplots(nax, nax)
	for i in range(nax):
		for j in range(nax):
			iy = int( i*NY/nax + NY/(2*nax) )
			ix = int( j*NX/nax + NX/(2*nax) )
			tit = "(" + str(xlocs[ix]) +"," + str(ylocs[iy]) + ")"
			axarr[i,j].set_title(tit)

			if i != nax-1:
				plt.setp(axarr[i,j].get_xticklabels(), visible=False)
			else:
				axarr[i,j].set_xlabel("ms")

			if j != 0:
				plt.setp(axarr[i,j].get_yticklabels(), visible=False)
			else:
				axarr[i,j].set_ylabel("uA")

			axarr[i,j].set_ylim(-.2,.6)
			axarr[i,j].plot(time,sIprobe[ix,iy,:])
#			axarr[i,j].plot(time,lines[ix,iy,:])

	plt.show()

#============================
if True:
	# save data
	ofn = r"P:\Data\Laptag\16-03-05 eBeam & GEA\plot_data"
	np.save(ofn, (sIprobe,time,xlocs,ylocs))  # writes file <ofn>.npy
	print("wrote file", ofn+".npy")
	print("now run plt_contours.py")
else:
	# do a contour plot of the results
	frst_ndx = int(0.87/dt)
	last_ndx = int(1.4/dt)

	for n in range(frst_ndx,last_ndx,20):
		t = time[n]
		plt.title("16-03-05")
		if n == frst_ndx:
			p = plt.imshow(sIprobe[:,:,n])  # first time through: generate plot
			plt.clim(0,.3)              # set color limits
			plt.colorbar()
		else:
			p.set_data(sIprobe[:,:,n])      # subsequently: change the data in the existing plot
		print(n, ": t = ", t)
		plt.pause(0.1)                   # shows latest version

	print("ok")
	plt_show()


print("done")
