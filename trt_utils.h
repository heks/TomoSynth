
#ifndef TRT_CUDA_SBDX_TRT_UTILS_H_
#define TRT_CUDA_SBDX_TRT_UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include "TomoSynth.h"


#define CUDA_CHECK_RETURN(value) {                                                     \
        cudaError_t _m_cudaStat = value;                                                \
        if (_m_cudaStat != cudaSuccess) {                                               \
                fprintf(stderr, "Error %s at line %d in file %s\n",                     \
                                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
                exit(1);                                                                \
        } }

// prototypes
/*
uInt llhread(
        char* fname,
        uInt whichframe,
        nova_str* nova,
        uInt8**** llhData
        );
        */
uInt8**** llhread(
        char* fname,
        uInt whichframe,
        nova_str* nova
        );

void *my_malloc ( char *expr, size_t size );
void my_free( void *ptr );

float **create_2D_float ( int max_x, int max_y );
uint32_t **create_2D_uint32_t(int max_x, int max_y);
float ****create_4D_float ( int max_x, int max_y, int max_r, int max_c );
uInt8 ****create_4D_uInt8 ( int max_x, int max_y, int max_r, int max_c );

int directoryExistsQ(const char * path);
char * directoryName(const char * path);
void createDirectory(const char * path);
 
#define MY_MALLOC(x)    my_malloc( #x, x )
#define MY_FREE(x)      my_free(x)




#endif /* TRT_CUDA_SBDX_TRT_UTILS_H_ */
