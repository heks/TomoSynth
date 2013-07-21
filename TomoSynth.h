
#ifndef TRT_CUDA_SBDX_TOMOSYNTH_H_
#define TRT_CUDA_SBDX_TOMOSYNTH_H_

#include <stdint.h>





// Convenient Data Types

typedef unsigned char uInt8;
typedef unsigned int uInt;

typedef struct nova_str {
    char hilf[32];
    uint32_t noimages;
    uint32_t detsizex;
    uint32_t detsizey;
    uint32_t row;
    uint32_t col;
    //uint32_t nofoc; // Not stored in header
    uint32_t noframe;
    uint32_t nosample;
} nova_str ;



#endif /* TOMOSYNTH_H_ */
