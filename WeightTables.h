
#ifndef TRT_CUDA_SBDX_WEIGHTTABLES_H_
#define TRT_CUDA_SBDX_WEIGHTTABLES_H_

#include "DetectorConfig.h"

// Fixed Device Parameters
#define M 10             // Pixel Span
#define M_HALF (M/2)
#define L_S 0.23 // (sourcepitch) Source pitch
#define D_SD 150.0 // (DSD) detector source distance
#define SPREAD 1


float**** initWeights(detector *d, float d_sp);

 __global__ void initWeightKernel(uint32_t sizedet1,uint32_t size0, uint32_t size1, float *x, float *y, float *z, float s, float *w );
//__global__ void initWeight2Kernel( uint32_t sizedet0, uint32_t sizedet1, float *nx, float *ny, float * w);

#endif /* WEIGHTTABLES_H_ */
