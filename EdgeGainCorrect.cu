
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "TomoSynth.h"
#include "WeightTables.h"

float* initEdgeGain(nova_str *nova , float **** weight , float ** nor)
{

    float *normim;
    int rpos, cpos;
    uint32_t r, c, i, j, w, idx;
    uint32_t Mr, Mc;
    Mr = M * nova->row;
    Mc = M * nova->col;
    const char* FNAME;
    FILE* fp;

    normim = (float *) calloc(Mr * Mc , sizeof(float)); //

    // Set image to zero
    /**
     for (r = 0; r < Mr; r++)
     {
     for (c = 0; c < Mc; c++)
     {
     reconim[r][c] = 1.0;
     }
     }
     */

    for(r = 0; r < nova->row; r++)
    {
        for(c = 0; c < nova->col; c++)
        {
            if (nor[r][c] > 0.0)
            {
                for(i = 0; i < nova->detsizex; i++)
                {

                    for(j = 0; j < nova->detsizey; j++)
                    {
                        for(w = 0; w < 4; w++)
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

                                    idx = rpos * Mc + cpos;
                                    //idx= rpos +Mc*cpos;
                                    // Compute a commulatvie pixel weight based on the interpolation weight and there mapped destination.
                                    normim[idx] = normim[idx]
                                            + weight[i][j][w][2];
                                    //}
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    FNAME = "run/normim.bin";
    fp = fopen(FNAME , "w");
    fwrite(normim , sizeof(float) , Mr * Mc , fp);
    fclose(fp);

    return normim;
}
