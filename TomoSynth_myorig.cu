
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

//#define verbose 1

#define SAVE_TIF 1

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char **argv)
{

    /* Parboil parameters */
    struct pb_Parameters *parameters;
    parameters = pb_ReadParameters(&argc, argv);

    /* Parboil timing */
    struct pb_TimerSet timers;
    pb_InitializeTimerSet(&timers);
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

    /* Variable declarations */
    FILE* fp;
    uInt8 ****llhData;
    float **nor = NULL;
    float ****w;
#ifdef GATHER
    float * reconim;
#else
    float ** reconim;
#endif
    nova_str* nova;
    nova = (nova_str *) malloc(sizeof(nova_str));

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
    printf("sizedet.x=%u, sizedet.y=%u\n" , d->sizedet[0] , d->sizedet[1]);

    /* Perform Dynamic Range Equalization */
    printf("\nPerform dynamic range equalization...\n");
    nor = DynRangeEqualization(nova , llhData);
    printf("nor[0][0]=%f\n" , nor[0][0]);
    printf("nova row=%d\n" , nova->row);
    printf("nova col=%d\n" , nova->col);
    
    /* Perform Reconstruction */
    float* d_sp;
    uint16_t z;
    d_sp = (float *) malloc(sizeof(float) * (NUM_PLANES + 1));
    d_sp[0] = D_SP; // read from file later;
    // d_sp[0] = 45;
    for(z = 0; z < NUM_PLANES; z++)
    {

        printf("\nReconstruction: processing plane %d out of %d...\n", z + 1, NUM_PLANES);

        printf("    initWeights\n");
        w = (float****) initWeights(d , d_sp[z]);

        printf("    Reconstruct\n");
        reconim = reconstruct(nova , llhData , w , nor);
        printf("    reconim[0][0]=%f\n" , reconim[0][0]);

        uint32_t Mr, Mc;

        Mr = M * nova->row;
        Mc = M * nova->col;

        const char* fname = "run/reconim.bin";

        // Save first plane as binary for debugging
        if (z == 0)
        {
            fp = fopen(fname , "w");
            numBytes = 0;

            uint32_t r, c;
            float* out;
            out = (float *) malloc(sizeof(float) * Mr * Mc);
            for(r = 0; r < Mr; r++)
            {
                for(c = 0; c < Mc; c++)
                {

                    numBytes += fwrite((float *) &reconim[r][c] ,
                            sizeof(float) , 1 , fp);
                    out[r * Mc + c] = (int16_t) reconim[r][c];
                }
            }

            printf("    numBytes written to reconim is %d\n" , numBytes);
            fclose(fp);
            free(out);
        }

#if SAVE_TIF
        const char* ext;
        char outname[100];
        char numstr[10];
        ext = ".tif";
        uint32_t r, c;
        float* out;
        out = (float *) malloc(sizeof(float) * Mr * Mc);
        for(r = 0; r < Mr; r++)
        {
            for(c = 0; c < Mc; c++)
            {
                out[r * Mc + c] = (int16_t) reconim[r][c];
            }
        }


        strcpy(outname , froot);
        sprintf(numstr , "/%3.4f" , numstr , d_sp[z]);
        strcat(outname , numstr);
        strcat(outname , ext);
	      printf("    writing to file %s\n", outname);
        writeTif(outname , (IP_PIMAGE) out , Mr , Mc);
        free(out);
#endif

        d_sp[z + 1] = d_sp[z] + DELTA_D_SP;
        
        printf("    end processing plane %d.\n", z);
        
    } // end for


    free(nova);
    free(d);
    MY_FREE( llhData[0][0][0]);
    MY_FREE( llhData[0][0]);
    MY_FREE( llhData[0]);
    MY_FREE( llhData);

    MY_FREE(nor[0]);
    MY_FREE(nor);

#ifdef GATHER
    MY_FREE(reconim);
#else
    MY_FREE(reconim[0]);
    MY_FREE(reconim);
#endif
    MY_FREE( w[0][0][0]);
    MY_FREE( w[0][0]);
    MY_FREE( w[0]);
    MY_FREE( w);

    /**


     CUDA_CHECK_RETURN(cudaFree((void*) d));
     CUDA_CHECK_RETURN(cudaDeviceReset());
     */


    pb_SwitchToTimer(&timers, pb_TimerID_NONE);
    pb_PrintTimerSet(&timers);
    pb_FreeParameters(parameters);
    
    return 0;
}


