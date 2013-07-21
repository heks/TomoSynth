
#include "TomoSynth.h"
#include "trt_utils.h"
#include "DynRangeEqual.h"

//__global__ float maxDet = 0.0;
//__global__ float maxNor;
//__global__ float tmp_sum = 0.0;
/*
__global__ void DynRangeKernel( uInt8 *llhData, uint32_t detsizex, uint32_t detsizey, uint32_t row, uint32_t col, float * maxNor, float *nor )
{
    int i, j, k;
    // r = x, c = y
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;

    float tmpDbg;
    float tmp_sum;
    float maxDet = 0.0;
    float nor_thresh;
    float p, s;
    uInt8 pixvalue;
    float norvalue = 0.0;

    //tmpDbg = 0.0;
    tmp_sum = 0.0;

    if (x < row && y < col)
    {
        for(i = 0; i < detsizex; i++)
        {
            for(j = 0; j < detsizey; j++)
            {   //i*detsizey*row*col+j*row*col+x*col+y
                pixvalue = llhData[i*detsizey*row*col+j*row*col+x*col+y];
                if (pixvalue > PIX_THRESH)
                    {
                        //printf("llhData[%i][%i][%i][%i]= %u, marked as bad\n" , i , j , x , y , pixvalue);
                        llhData[i*detsizey*row*col+j*row*col+x*col+y] = 0;
                        pixvalue = 0;
                    }
                //for(k = 0; k < detsizex * detsizey; k++)
                tmp_sum += llhData[i*detsizey*row*col+j*row*col+1*col+1];

                //tmpDbg += pixvalue;
                norvalue += pixvalue;
                //nor[x*col+y] += pixvalue;

                if( pixvalue > (maxDet) ){
                    (maxDet) = pixvalue;
                }

                if (norvalue > 0){
                    p = (maxDet) / norvalue;
                    if (p > (*maxNor)){
                        (*maxNor) = p;
                    }
                }
            }
        }
    }
}
__global__ void DynRangeKernel( uInt8 *llhData, uint32_t detsizex, uint32_t detsizey, uint32_t row, uint32_t col, float * maxNor, float *nor )
{
        nor_thresh = NOR_SCALE * tmp_sum / (detsizex * detsizey);
    if (x < row && y < col)
    {
        if (norvalue > nor_thresh)
        {
            if (DRE == 1) // If dynamic Range Equalization is enabled, do it.
            {
                s = PIX_MAX / (norvalue * (*maxNor));

                for(i = 0; i < detsizex; i++)
                {
                    for(j = 0; j < detsizey; j++)
                    {
                        llhData[i*detsizey*row*col+j*row*col+x*col+y] = (uint8_t) (s * llhData[i*detsizey*row*col+j*row*col+x*col+y]);
                    }
                }
            }
        }
        else // If not doing DRE, zero out bad pixels
        {
            norvalue = 0.0;
            for(i = 0; i < detsizex; i++)
            {
                for(j = 0; j < detsizey; j++)
                {
                    llhData[i*detsizey*row*col+j*row*col+x*col+y] = 0;
                }
            }
        }
    }

    nor[x*col+y] = norvalue;
}
*/


float** DynRangeEqualization(nova_str *nova , uint8_t ****llhData)
{
    int r, c, j, k;
    float **nor;
    float tmpDbg, tmp_sum;
    float nor_thresh;
    float maxDet,maxNor;
    float p;
    nor = create_2D_float(nova->col , nova->row);


    //thr=sum(sum(llhData(:,:,2,2),1),2)/(sizedet(1)*sizedet(2))*1.05;
    tmp_sum = 0.0;
    for(j = 0; j < nova->detsizex; j++)
    {
        for(k = 0; k < nova->detsizey; k++)
        {
            for(r = 0; r < nova->row; r++)
            {
                for(c = 0; c < nova->col; c++)
                {

                    if (llhData[j][k][r][c] > PIX_THRESH)
                    { // Remove Bad Pixels
                        printf("llhData[%i][%i][%i][%i]= %u, marked as bad\n" ,
                                j , k , r , c , llhData[j][k][r][c]);
                        llhData[j][k][r][c] = 0;
                    }
           //         tmp_sum += llhData[j][k][1][1];
                }
            }
            tmp_sum += llhData[j][k][1][1];
        }
    }
    nor_thresh = NOR_SCALE * tmp_sum / (nova->detsizex * nova->detsizey);
    //nor_thresh=0.6;

    for(r = 0; r < nova->row; r++)
    {
        for(c = 0; c < nova->col; c++)
        {
            // Compute Mean
            nor[r][c] = 0.0;
            tmpDbg = 0.0;
            maxDet=0.0;
            for(j = 0; j < nova->detsizex; j++)
            {
                for(k = 0; k < nova->detsizey; k++)
                {

                    tmpDbg += llhData[j][k][r][c];
                    nor[r][c] += llhData[j][k][r][c];
                    if( llhData[j][k][r][c]>maxDet){
                    		maxDet=llhData[j][k][r][c];
                    }
                }
            }

            if (nor[r][c]>0){
            	p=maxDet/nor[r][c];
            	if (p > maxNor){
            		maxNor=p;
            	}
            }
        }
    }
     //       nor[r][c] /= (float) (nova->detsizex * nova->detsizey);

            float s;
            // Equalize

    for(r = 0; r < nova->row; r++)
    {
        for(c = 0; c < nova->col; c++)
        {
            if (nor[r][c] > nor_thresh)
            {
                if (DRE == 1) // If dynamic Range Equalization is enabled, do it.
                {
                    s = PIX_MAX / (nor[r][c] * maxNor);
//                    printf("s=%f\n" , s);
                    for(j = 0; j < nova->detsizex; j++)
                    {
                        for(k = 0; k < nova->detsizey; k++)
                        {

                            llhData[j][k][r][c] = (uint8_t) (s
                                    * llhData[j][k][r][c]);

                        }
                    }
                }
            }
            else // If not doing DRE, zero out bad pixels
            {
                nor[r][c] = 0.0;
                for(j = 0; j < nova->detsizex; j++)
                {
                    for(k = 0; k < nova->detsizey; k++)
                    {
                        llhData[j][k][r][c] = 0;
                    }
                }
            }
        }

    }
#ifdef VERBOSE
    for (r = 0; r < nova->row; r++)
    {
        for (c = 0; c < nova->col; c++)
        {
            printf("nor[%i][%i]=%f ",r,c,nor[r][c]);
        }
        printf("\n");
    }
#endif

    return nor;
}

