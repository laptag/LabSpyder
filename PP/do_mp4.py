import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_animation

#============================================================================
import scipy.ndimage as synd

def smooth(data, interval):
	return synd.filters.uniform_filter(data, interval)

#============================================================================

ifn = r"P:\Data\Laptag\16-03-05 eBeam & GEA\plot_data.npy"

print("Reading file", ifn)

(sIprobe,time,xlocs,ylocs) = np.load(ifn)  # reads file <ofn>.npy

NTimes = np.size(time)
NX     = np.size(xlocs)
NY     = np.size(ylocs)
dt     = time[1]-time[0]   # ms

##############   ffmpeg binaries copied to "c:\program files\anaconda"  (already in path) #################

FFMpegWriter = mpl_animation.writers['ffmpeg']
metadata = dict(title='16-03-05', artist='laptag,pp', comment='beam scattered')
writer = FFMpegWriter(fps=15, metadata=metadata, extra_args=("-crf","0"))

fig = plt.figure()
plt.title("16-03-05")
plt.xlabel("mm")
plt.ylabel("mm")

mfn = r"P:\Data\Laptag\16-03-05 eBeam & GEA\16-03-08.mp4"

frst_ndx = int(0.9/dt)
last_ndx = int(1.4/dt)

data=sIprobe[:,:,frst_ndx]  #np.zeros((41,41))

p = plt.imshow(data)
plt.clim(0,.5)              # set color limits
plt.colorbar()

with writer.saving(fig, mfn, 100):
	for i in range(frst_ndx,last_ndx,5):
		data = sIprobe[:,:,i]
		p.set_data(smooth(data,3))
		#p.set_data(data)
		tstr = str(time[i])+" ms"
		print(tstr)
		plt.text(x=26, y=-1.3, s=tstr, backgroundcolor='#eeefff')
		writer.grab_frame()

print("done")
