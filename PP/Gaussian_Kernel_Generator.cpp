// Gaussian_Kernel_Generator.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "math.h"
#include <stdio.h>

#ifdef __cplusplus    // If used by C++ code, 
extern "C" {          // we need to export the C interface
#endif

inline double sq(double x) { return x*x; }

__declspec(dllexport) int Gaussian_Kernel_Generator(size_t data_size, size_t width, double* data)
{	double sum = 0;
	for(size_t i = 0; i < data_size; ++i)
	{	data[i] = exp(-sq((i-data_size/2.)/double(width)));
		sum += data[i];
	}
	for(size_t i = 0; i < data_size; ++i)
		data[i] /= sum;

	return 0;
}

#ifdef __cplusplus
}
#endif
