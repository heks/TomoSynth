 
#ifndef RECONSTRUCT_H_
#define RECONSTRUCT_H_

#define ANGLE_DEGREES (-1.426)
#define PI 3.141592653589793
#define DROTATION ANGLE_DEGREES*PI/180.0
#define NUM_PLANES 96
#define DELTA_D_SP 0.2

#define D_SP 35.4  // (plane_z) source to focal-plane distance



void  reconstruct(nova_str* nova, uInt8**** llhData, float**** w, float** nor, uInt8 *llhData_d, float **reconim, float * weight_d, float * nor_d, float *normin);
float * reconstruct_gather(nova_str *nova , detector* d , uInt8 ****llhData , float **** weight , float ** nor, float d_sp);
__global__ void reconKernel(uInt8 *llhData, float * weight, float *reconim, float *nor, uint32_t detsizex, uint32_t detsizey, uint32_t row, uint32_t col, float *normin);
__global__ void edgeGainKernel(uint32_t Mrow, uint32_t Mcol, float *reconim, float *normim );
#endif /* RECONSTRUCT_H_ */



    