
#ifndef DYNRANGEEQUAL_H_
#define DYNRANGEEQUAL_H_

//#define NOR_SCALE 1.05

#define PIX_THRESH 34
#define PIX_MAX 255

#define DRE 1 // 1 turns dynamic range equaliztion on

#if DRE

    #define NOR_SCALE 1.05
#else
    #define NOR_SCALE 1.05
#endif


float** DynRangeEqualization(nova_str *nova, uint8_t ****llhData);
__global__ void DynRangeKernel( uInt8 *llhData, uint32_t detsizex, uint32_t detsizey, uint32_t row, uint32_t col, float * maxNor, float *nor );
#endif /* DYNRANGEEQUAL_H_ */
