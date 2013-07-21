
#include "TomoSynth.h"

#include "TomoSynth.h"
#include "WeightTables.h"
#include "trt_utils.h"
#include "EdgeGainCorrect.h"
#include <stddef.h>
#include "Reconstruct.h"
#include "SensorInfo.h"


#define TILE_SIZE 8
#define CUDACheck(stmt) do {\
        cudaError_t err = stmt;\
        if (err != cudaSuccess) {\
            printf("ERROR: Failed to run %s on line %d in function %s.\n", #stmt, __LINE__, __func__);    \
            exit(-1);  \
        }\
    } while(0)


__global__ void reconKernel( uInt8 *llhData, float * weight, float *reconim, float *nor, uint32_t detsizex, uint32_t detsizey, uint32_t row, uint32_t col, float *normim)
{
    int rpos, cpos;
    int i, j;
    int idx;

    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < row && y < col)
    {
        if( nor[x*col+y] > 0.0 )
        {
            for (i = 0; i < detsizex; i++)
            {
                for (j = 0; j < detsizey; j++)
                {
                //  for (w = 0; w < 4; w++)
                //  {
                        rpos = (weight[i*detsizey*4*3+j*4*3+w*3+0]) + (x * M) + M_HALF;

                        if ((rpos > 0) && (rpos <= M * row))

                        {
                            cpos = (weight[i*detsizey*4*3+j*4*3+w*3+1]) + (y * M)+ M_HALF;

                            if ((cpos > 0 && (cpos <= M * col)))
                            {
                                rpos = rpos - 1;
                                cpos = cpos - 1;
                                idx = rpos * M * col + cpos; 

                                atomicAdd(&reconim[col*M*rpos+cpos], llhData[i*detsizey*row*col+j*row*col+x*col+y] * weight[i*detsizey*4*3+j*4*3+w*3+2]);

                                atomicAdd(&normim[idx], weight[i*detsizey*4*3+j*4*3+w*3+2]); 

                            }
                        }
                //  }
                }
            }
        }
    }

}

__global__ void edgeGainKernel( uint32_t Mrow, uint32_t Mcol, float *reconim, float *normim ){



    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    int idx;

    idx = r * Mcol + c;
    if (r < Mrow && c < Mcol && normim[idx] > 0.0)
        reconim[ r*Mcol+c] = reconim[r*Mcol+c] / normim[idx];
 

}


//cpu reconstruct

float** reconstruct(nova_str *nova, uInt8 ****llhData, float **** weight,
        float ** nor)
{

    float **reconim;
    int rpos, cpos;
    int r, c, i, j, w;
    uint32_t Mr, Mc;
    Mr = M * nova->row;
    Mc = M * nova->col;


    reconim = (float **) create_2D_float(Mr, Mc);

    printf("        M=%i\n", M);

    // Set image to zero
    for (r = 0; r < Mr; r++)
    {
        for (c = 0; c < Mc; c++)
        {
            reconim[r][c] = 0.0;
        }
    }

    int idx;
    // Begin reconstruction
    for (r = 0; r < nova->row; r++)
    {
        for (c = 0; c < nova->col; c++)
        {
            if (nor[r][c] > 0.0)
            {
                for (i = 0; i < nova->detsizex; i++)
                {

                    for (j = 0; j < nova->detsizey; j++)
                    {
                        for (w = 0; w < 4; w++)
                        {

                            rpos = (weight[i][j][w][0]) + (r) * (M) + M_HALF;

                            if ((rpos > 0) && (rpos <= M * nova->row))

                            {
                                cpos =
                                        (weight[i][j][w][1])
                                                + (c) * (M)+ M_HALF;

                                if ((cpos > 0 && (cpos <= M * nova->col)))
                                {
                                    rpos = rpos - 1;
                                    cpos = cpos - 1;

#ifdef VERBOSE
                                    printf("%i %i %i %i 0 %2.4e %i %i\n",rpos, i, j, w, weight[i][j][w][0],r, M*r+M_HALF);
                                    printf("%i %i %i %i 1 %2.4e %i %i\n",cpos, i, j, w, weight[i][j][w][1],c, M*c+M_HALF);
                                    printf("%i  \n\n",llhData[i][j][c][r]);
#endif
                                    //if (llhData[i][j][r][c] >0 ){

                                    reconim[rpos][cpos] = reconim[rpos][cpos]
                                            + llhData[i][j][r][c]
                                                    * weight[i][j][w][2];

                                    //}
                                }
                            }
                        }

                    }
                }
            }
        }
    }

#if EDGE_GAIN_CORRECTION

    float* normim;

    normim = initEdgeGain(nova, weight, nor);

    for (r = 0; r < Mr; r++)
    {
        for (c = 0; c < Mc; c++)
        {
            idx = r * Mc + c;
            //idx=r+c*Mc;
            if (normim[idx] > 0.0)
            {
                reconim[r][c] = reconim[r][c] / normim[idx];
            }
        }
    }

    free(normim);
#endif

    return reconim;

}

