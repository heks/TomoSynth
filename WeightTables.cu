
#include <stdint.h>
#include <stdio.h>
#include "DetectorConfig.h"
#include "WeightTables.h"
#include "trt_utils.h"


 __global__ void initWeightKernel( uint32_t sizedet1, uint32_t size0, uint32_t size1, float *x, float *y, float *z, float s, float *w ){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i < size0  && j < size1)
    {
    
		float dx, dy;
		float temp_nx,temp_ny;

		/* intermediate step*/
		temp_nx = (float)(s * x[i*size1+j] / z[i*size1+j]);
		temp_ny = (float)(s * y[i*size1+j] / z[i*size1+j]);

		float wx[4];
		float wy[4];

        if(SPREAD == 0){
            dx = 0.0;
            dy = 0.0;
        }
        else{
            dx = temp_nx - floorf(temp_nx);
            dy = temp_ny - floorf(temp_ny);
        }
        wx[0] = floorf(temp_nx) - 1;
        wx[1] = 1 - dx;
        wx[2] = floorf(temp_nx);
        wx[3] = dx;

        wy[0] = floorf(temp_ny) - 1;
        wy[1] = 1 - dy;
        wy[2] = floorf(temp_ny);
        wy[3] = dy;

        w[i*sizedet1*4*3+j*4*3+3*0+0] = wx[0];
        w[i*sizedet1*4*3+j*4*3+3*0+1] = wy[0];
        w[i*sizedet1*4*3+j*4*3+3*0+2] = wx[1] * wy[1];

        w[i*sizedet1*4*3+j*4*3+3*1+0] = wx[0];
        w[i*sizedet1*4*3+j*4*3+3*1+1] = wy[2];
        w[i*sizedet1*4*3+j*4*3+3*1+2] = wx[1] * wy[3];

        w[i*sizedet1*4*3+j*4*3+3*2+0] = wx[2];
        w[i*sizedet1*4*3+j*4*3+3*2+1] = wy[0];
        w[i*sizedet1*4*3+j*4*3+3*2+2] = wx[3] * wy[1];

        w[i*sizedet1*4*3+j*4*3+3*3+0] = wx[2];
        w[i*sizedet1*4*3+j*4*3+3*3+1] = wy[2];
        w[i*sizedet1*4*3+j*4*3+3*3+2] = wx[3] * wy[3];
     }
}

float**** initWeights(detector *d, float d_sp)
{

    printf("        d_sp= %f\n", d_sp);
    uint16_t i, j;
    float **nx, **ny;
    //uint32_t s;
    float s;

    int r = d->sizedet[0];
    int c = d->sizedet[1];
    float **x = d->x_det;
    float **y = d->y_det;
    float **z = d->z_det;
    float **wx, **wy;
    float ****w;
    float dx, dy;
    double L_P;

    L_P= (L_S/M/D_SD*(D_SD-d_sp)); //V
    s = (float )(((d->pxlSize) * d_sp )/ L_P); // Pre-calculate pixelsize*imagepix/plane_z

    printf("        pxlSize= %f\n",d->pxlSize);
    printf("        d_sp=%f\n",d_sp);
    printf("        L_P=%f\n",L_P);
    printf("        s=%f\n",s);

    nx = create_2D_float(r, c);
    ny = create_2D_float(r, c);

#if 0
    printf("ny \n");
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            nx[i][j] = (float) (s * x[i * c + j] / z[i * c + j]);
            ny[i][j] = (float) (s * y[i * c + j] / z[i * c + j]);
#ifdef VERBOSE
            printf("%f ",ny[i][j]);
#endif
        }
#ifdef VERBOSE
        printf("\n");
#endif
    }
#endif

    for (i = 0; i < r; i++)
        {
            for (j = 0; j < c; j++)
            {
                nx[i][j] = (float) (s * x[i][j] / z[i][j]);
                ny[i][j] = (float) (s * y[i][j] / z[i][j]);
    #ifdef VERBOSE
                printf("%f ",ny[i][j]);
    #endif
            }
    #ifdef VERBOSE
            printf("\n");
    #endif
        }


    //n=pixelsize/150*plane_z/imagpix;
    //n=L_P
    /**
     * \todo need to make this more general
     */
    wx = create_2D_float(2, 2);
    wy = create_2D_float(2, 2);
    w = create_4D_float(r, c, 4, 3);
    // int n = (int) s /D_SD;


    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            if (SPREAD == 0)
            {
                dx = 0.0;
            }
            else
            {
                dx = nx[i][j] - floorf(nx[i][j]);
            }
            wx[0][0] = floorf(nx[i][j]) - 1;
            wx[0][1] = 1 - dx;
            wx[1][0] = floorf(nx[i][j]);
            wx[1][1] = dx;

            if (SPREAD == 0)
            {
                dy = 0.0;
            }
            else
            {
                dy = ny[i][j] - floorf(ny[i][j]);
            }
            wy[0][0] = floorf(ny[i][j]) - 1;
            wy[0][1] = 1                - dy;
            wy[1][0] = floorf(ny[i][j]);
            wy[1][1] = dy;

            w[i][j][0][0] = wx[0][0];
            w[i][j][0][1] = wy[0][0];
            w[i][j][0][2] = wx[0][1] * wy[0][1];

            w[i][j][1][0] = wx[0][0];
            w[i][j][1][1] = wy[1][0];
            w[i][j][1][2] = wx[0][1] * wy[1][1];

            w[i][j][2][0] = wx[1][0];
            w[i][j][2][1] = wy[0][0];
            w[i][j][2][2] = wx[1][1] * wy[0][1];

            w[i][j][3][0] = wx[1][0];
            w[i][j][3][1] = wy[1][0];
            w[i][j][3][2] = wx[1][1] * wy[1][1];

        }
    }

    MY_FREE( nx[0]);
    MY_FREE( nx);

    MY_FREE( ny[0]);
    MY_FREE( ny);
    MY_FREE(wx[0]);
    MY_FREE(wx);
    MY_FREE(wy[0]);
    MY_FREE(wy);

    return w;
}

