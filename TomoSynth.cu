
#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>

#include "TomoSynth.h"
#include "trt_utils.h"
#include "DetectorConfig.h"
#include "WeightTables.h"
#include "DynRangeEqual.h"
#include "Reconstruct.h"
#include "writeTif.h"
#include "stdHdr.h"

#define CUDACheck(stmt) do {\
        cudaError_t err = stmt;\
        if (err != cudaSuccess) {\
            printf("ERROR: Failed to run %s on line %d in function %s.\n", #stmt, __LINE__, __func__);    \
            exit(-1);  \
        }\
    } while(0)

#define TILE_SIZE 8

//#define verbose 1

#define SAVE_TIF 1

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv)
{
    /* Host Variables */
    uInt8 ****llhData;
    float **nor = NULL;
    nova_str* nova;
    nova = (nova_str *) malloc(sizeof(nova_str));
    float* out;

    /* Device Variables */
    uInt8 *llhData_d;
    float * reconim_d;
    float * normin_d;
    float * weight_d;
    float * nor_d;
    float * detx_d;
    float * dety_d;
    float * detz_d;

    /* Parboil parameters */
    struct pb_Parameters *parameters;
    parameters = pb_ReadParameters(&argc, argv);

    /* Parboil timing */
    struct pb_TimerSet timers;
    pb_InitializeTimerSet(&timers);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    /* Variable declarations */
    float s;
    double L_P;
    const char* ext;
    ext = ".tif";
    char outname[100];
    char numstr[10];

    /* Read Input File */
    printf("\nRead input file...\n");
    const char *llhFile = parameters->inpFiles[0];
    uint32_t whichframe = 1;
    llhData = llhread((char *) llhFile, whichframe , nova);
    const char * froot = directoryName(parameters->outFile);
    createDirectory(froot);

    /* Read Detector configuration */
    printf("\nRead detector configuration...\n");
    struct detector *d = (struct detector *) malloc(sizeof(struct detector));
    char *xFile = parameters->inpFiles[1];
    char *yFile = parameters->inpFiles[2];
    char *zFile = parameters->inpFiles[3];
    uint32_t numBytes = DetectorConfig(d, xFile, yFile, zFile);
    
    //Initialize grid dimensions
    uint32_t Mr, Mc;
    Mr = M * nova->row;
    Mc = M * nova->col;

    dim3 gridDim( ceil( (float) nova->row / (float) TILE_SIZE ) , ceil( (float) nova->col / (float) TILE_SIZE ) , 4 );
    dim3 edgegridDim( ceil( (float) Mr / (float) TILE_SIZE ), ceil( (float) Mc / (float) TILE_SIZE ) , 1);
    dim3 initWeightgridDim( ceil( (float) d->sizedet[0] / (float) TILE_SIZE ), ceil( (float) d->sizedet[1] / (float) TILE_SIZE ) , 1);
    dim3 blockDim( TILE_SIZE, TILE_SIZE, 1 );

    /* Perform Reconstruction */
    float* d_sp;
    uint16_t z;
    d_sp = (float *) malloc(sizeof(float) * (NUM_PLANES + 1));
    d_sp[0] = D_SP; // read from file later;

    out = (float *) malloc(sizeof(float) * Mr * Mc);

    // Device allocate
    cudaMalloc( (void **) &llhData_d, nova->row * nova->col * nova->detsizex * nova->detsizey * sizeof(uInt8) );
    cudaMalloc( (void **) &nor_d, nova->row * nova->col * sizeof(float) );
    cudaMalloc( (void **) &weight_d, nova->detsizex * nova->detsizey * 4 * 3 * sizeof(float) );
    cudaMalloc( (void **) &reconim_d, Mr * Mc * sizeof(float) );
    cudaMalloc( (void **) &normin_d, Mr * Mc * sizeof(float) );

    cudaMalloc( (void **) &detx_d, d->sizedet[0]*d->sizedet[1] * sizeof(float) );
    cudaMalloc( (void **) &dety_d, d->sizedet[0]*d->sizedet[1] * sizeof(float) );
    cudaMalloc( (void **) &detz_d, d->sizedet[0]*d->sizedet[1] * sizeof(float) );

    /* Perform Dynamic Range Equalization */
    //printf("\nPerform dynamic range equalization...\n");
    //reconKernel <<< gridDim, blockDim >>> (llhData_d, weight_d, reconim_d, nor_d, nova->detsizex, nova->detsizey, nova->row, nova->col, normin_d);
    //DynRangeKernel <<< gridDim, blockDim >>> ( llhData_d, nova->detsizex, nova->detsizey, nova->row, nova->col, maxNor_d, nor_d );
    //DynEqualizeKernel <<< gridDim, blockDim >>> 
    nor = DynRangeEqualization(nova , llhData);
    //printf("nor[0][0]=%f\n" , nor[0][0]);

    //Copy to Device
    cudaMemcpy( llhData_d, ***llhData, nova->row * nova->col * nova->detsizex * nova->detsizey * sizeof(uInt8), cudaMemcpyHostToDevice );
    cudaMemcpy( nor_d, *nor, nova->col * nova->row * sizeof(float), cudaMemcpyHostToDevice );

    cudaMemcpy( detx_d, *d->x_det, d->sizedet[0]*d->sizedet[1] * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dety_d, *d->y_det, d->sizedet[0]*d->sizedet[1] * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( detz_d, *d->z_det, d->sizedet[0]*d->sizedet[1] * sizeof(float), cudaMemcpyHostToDevice );

    /***********************************************NUMPLANES****************************************************/
    for(z = 0; z < NUM_PLANES; z++) 
    {
       
        //printf("\nReconstruction: processing plane %d out of %d...\n", z + 1, NUM_PLANES);

        //Initialize Weights
        L_P= (L_S/M/D_SD*(D_SD-d_sp[z]));
        s = (float)(d->pxlSize * d_sp[z] / L_P);

        initWeightKernel <<< initWeightgridDim, blockDim >>> (nova->detsizey,d->sizedet[0],d->sizedet[1],detx_d, dety_d, detz_d, s, weight_d);

        //Reconstruction
        cudaMemset(reconim_d, 0.0, Mr * Mc * sizeof(float));
        cudaMemset(normin_d, 0.0, Mr * Mc * sizeof(float));

        reconKernel <<< gridDim, blockDim >>> (llhData_d, weight_d, reconim_d, nor_d, nova->detsizex, nova->detsizey, nova->row, nova->col, normin_d);
        edgeGainKernel <<< edgegridDim, blockDim >>> (Mr, Mc, reconim_d, normin_d);                                                        

        //Copy back to host/file out
        cudaMemcpy(out, reconim_d, Mr * Mc * sizeof(float), cudaMemcpyDeviceToHost);

        strcpy(outname , froot);
        sprintf(numstr , "/%3.4f" , numstr , d_sp[z]);
        strcat(outname , numstr);
        strcat(outname , ext);
        //printf("    Writing to file %s\n", outname);
        writeTif(outname , (IP_PIMAGE) out , Mr , Mc);

        d_sp[z + 1] = d_sp[z] + DELTA_D_SP;
        
    }

    //Free memory
    cudaFree(weight_d);
    cudaFree(reconim_d);
    cudaFree(nor_d);
    cudaFree(normin_d);
    cudaFree(llhData_d);

    cudaFree(detx_d);
    cudaFree(dety_d);
    cudaFree(detz_d);

    free(out);
    free(nova);
    free(d);

    MY_FREE( llhData[0][0][0]);
    MY_FREE( llhData[0][0]);
    MY_FREE( llhData[0]);
    MY_FREE( llhData);

    MY_FREE(nor[0]);
    MY_FREE(nor);


    pb_SwitchToTimer(&timers, pb_TimerID_NONE);
    pb_PrintTimerSet(&timers);
    pb_FreeParameters(parameters);
    
    return 0;
}


