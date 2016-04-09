# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as sysi

def gsmooth(data, smooth_interval):

	num_vals = np.size(data)

	import ctypes
	# use the Gaussian_Kernel_Generator function from the eponymous DLL:
	dllfn = r'P:\P\DLLs\Gaussian_Kernel_Generator\x64\Release\Gaussian_Kernel_Generator.dll'
	gkg = ctypes.WinDLL(dllfn, use_last_error=True)
	e = ctypes.get_last_error()
	if e != 0:
		raise ctypes.WinError(e)

	gkg.Gaussian_Kernel_Generator.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_double)]
	gkg.Gaussian_Kernel_Generator.restype  = ctypes.c_int

	# set up args
	gen_data  = (ctypes.c_double*num_vals)()     # ??? arggh what does this even mean ???...--> allocate space
	data_size = ctypes.c_size_t(num_vals)
	width     = ctypes.c_size_t(smooth_interval)

	# generate the data
	gkg.Gaussian_Kernel_Generator(data_size, width, gen_data)
	e = ctypes.get_last_error()

	# unload the dll loaded above:
	from ctypes.wintypes import HMODULE
	ctypes.windll.kernel32.FreeLibrary.argtypes = [HMODULE]
	ctypes.windll.kernel32.FreeLibrary(gkg._handle)
	del gkg

	if e != 0:
		raise ctypes.WinError(e)

	# copy the data (again...):
	kernel = np.empty(num_vals, dtype=np.float)
	for i in range(num_vals):
		kernel[i] = gen_data[i]

	return sysi.fftconvolve(kernel, data, mode='same')	  # Mode 'same' returns output of length max(M, N). Boundary effects are still visible.


def test_gsmooth(num_pts=1000000, npts=20000):
	import pylab as plt
	data = np.ndarray(shape=(num_pts), dtype=np.float)
	for i in range(num_pts):
		if i < num_pts/2:
			data[i] = 0;
		else:
			data[i] = 1;
	plt.plot(data)
	sdata = gsmooth(data, npts)
	plt.plot(sdata)
	plt.show()
