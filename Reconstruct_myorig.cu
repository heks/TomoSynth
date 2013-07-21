
#include "TomoSynth.h"

#include "TomoSynth.h"
#include "WeightTables.h"
#include "trt_utils.h"
#include "EdgeGainCorrect.h"
#include <stddef.h>
#include "Reconstruct.h"
#include "SensorInfo.h"

#define TILE_SIZE 8

__global__ void reconKernel(uInt8 ****llhData, float **** weight, float **reconim, float ** nor, int detsizex, int detsizey, int row, int col)
{
	int rpos, cpos;
	int i, j, w;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < row && y < col)
	{
		if( nor[x][y] > 0.0 )
		{
			for (i = 0; i < detsizex; i++)
			{
				for (j = 0; j < detsizey; j++)
				{
					for (w = 0; w < 4; w++)
					{
						rpos = (weight[i][j][w][0]) + (x) * (M) + M_HALF;

						if ((rpos > 0) && (rpos <= M * row))

						{
							cpos = (weight[i][j][w][1]) + (y) * (M)+ M_HALF;

							if ((cpos > 0 && (cpos <= M * col)))
							{
								rpos = rpos - 1;
								cpos = cpos - 1;

								reconim[rpos][cpos] = reconim[rpos][cpos] + llhData[i][j][x][y] * weight[i][j][w][2];

							}
						}
					}
				}
			}
		}
	}

/*
	if ( x < nova->detsizex && y < nova->detsizey )
	{
		for (w = 0; w < 4; w++)
		{
			rpos = (weight[x][y][w][0]) + (r) * (M) + M_HALF;

			if ((rpos > 0) && (rpos <= M * nova->row))
			{
				cpos = (weight[x][y][w][1]) + (c) * (M)+ M_HALF;

				if ((cpos > 0 && (cpos <= M * nova->col)))
				{
					rpos = rpos - 1;
					cpos = cpos - 1;

					reconim[rpos][cpos] = reconim[rpos][cpos] + llhData[x][y][r][c] * weight[x][y][w][2];
				}
			}
		}
	}
*/
}

float** reconstruct(nova_str *nova, uInt8 ****llhData, float **** weight, float ** nor)
{

	float **reconim;
	int rpos, cpos;
	int r, c, i, j, w;
	uint32_t Mr, Mc;
	Mr = M * nova->row;
	Mc = M * nova->col;

	uInt8 ****llhData_d;
	float **reconim_d;
    float **** weight_d;
    float ** nor_d;

	reconim = (float **) create_2D_float(Mr, Mc);

	//cudaMalloc( (void *****) &llhData_d, M * nova->row * M * nova->col * sizeof(uInt8) );
	cudaMalloc( (void *****) &llhData_d, 68724272 );
    cudaMalloc( (void *****) &weight_d, nova->detsizex * nova->detsizey * 4 * 3 * sizeof(float) );
    
    //cudaMalloc( (void *****) &weight_d, M * M * 4 * 3 * sizeof(float) );
    cudaMalloc( (void ***) &nor_d, nova->row * nova->col * sizeof(float) );
	cudaMalloc( (void ***) &reconim_d, Mr * Mc * sizeof(float) );

	printf("        M=%i\n", M);

	// Set image to zero
	for (r = 0; r < Mr; r++)
	{
		for (c = 0; c < Mc; c++)
		{
			reconim[r][c] = 0.0;
		}
	}

	//cudaMemcpy( llhData_d, llhData, M * nova->row * M * nova->col * sizeof(uInt8), cudaMemcpyHostToDevice );
	cudaMemcpy( llhData_d, llhData, 68724272, cudaMemcpyHostToDevice );
    cudaMemcpy( weight_d, weight, nova->detsizex * nova->detsizey * 4 * 3 * sizeof(float), cudaMemcpyHostToDevice );
    //cudaMemcpy( weight_d, weight, M * M * 4 * 3 * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( nor_d, nor, nova->row * nova->col * sizeof(float), cudaMemcpyHostToDevice );
	cudaMemcpy( reconim_d, reconim, Mr * Mc * sizeof(float), cudaMemcpyHostToDevice );

    dim3 gridDim( nova->col / TILE_SIZE + 1, nova->row / TILE_SIZE + 1, 1 );
    dim3 blockDim( TILE_SIZE, TILE_SIZE, 1 );

	int idx;
	// Begin reconstruction

	reconKernel <<< gridDim, blockDim >>> (llhData_d, weight_d, reconim_d, nor_d, nova->detsizex, nova->detsizey, nova->row, nova->col);


	cudaMemcpy(reconim, reconim_d, Mr * Mc * sizeof(float), cudaMemcpyDeviceToHost);
/*
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
*/

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
	cudaFree( &llhData_d ); 
	cudaFree( &weight_d );
	cudaFree( &reconim_d );
	free(normim);
#endif

	return reconim;

}

