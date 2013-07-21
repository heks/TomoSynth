
#include <stdio.h>
#include <stdlib.h>

#include "TomoSynth.h"
#include "trt_utils.h"

#include "sys/stat.h"
#include "dirent.h"
#include "libgen.h"

#define VERBOSE 0
#define ALL_ONES 0


/***************************************
 http://cboard.cprogramming.com/c-programming/61578-4-dimensional-array-contiguous-allocation.html#post438210
 */

void *my_malloc(char *expr, size_t size)
{
	void *result = malloc(size);
#if VERBOSE
	printf("Malloc(%s) is size %lu, returning %p\n", expr, (unsigned long) size,
			result);
#endif
	return result;
}
void my_free(void *ptr)
{
#if VERBOSE
	printf("Free(%p)\n", ptr);
#endif
	free(ptr);
}

/* create float [x][y] */
float **create_2D_float(int max_x, int max_y)
{
	float **all_x = (float **) MY_MALLOC( max_x * sizeof *all_x );
	float *all_y = (float *) MY_MALLOC( max_x * max_y * sizeof *all_y );
	float **result = all_x;
	int x;

	for (x = 0; x < max_x; x++, all_y += max_y)
	{
		result[x] = all_y;
	}

	return result;
}

/* create float [x][y] */
uint32_t **create_2D_uint32_t(int max_x, int max_y)
{
    uint32_t **all_x = (uint32_t **) MY_MALLOC( max_x * sizeof *all_x );
    uint32_t *all_y = (uint32_t *) MY_MALLOC( max_x * max_y * sizeof *all_y );
    uint32_t **result = all_x;
    int x;

    for (x = 0; x < max_x; x++, all_y += max_y)
    {
        result[x] = all_y;
    }

    return result;
}
float ****create_4D_float(int max_x, int max_y, int max_r, int max_c)
{
	float ****all_x = (float ****) MY_MALLOC( max_x * sizeof (*all_x) );
	float ***all_y = (float ***) MY_MALLOC( max_x * max_y * sizeof( *all_y) );
	float **all_r =
			(float **) MY_MALLOC( max_x * max_y * max_r * sizeof( *all_r ));
	float *all_c =
			(float *) MY_MALLOC( max_x * max_y * max_r * max_c * sizeof(*all_c) );
	float ****result = all_x;
	int x, y, r;

	for (x = 0; x < max_x; x++, all_y += max_y)
	{
		result[x] = all_y;
		for (y = 0; y < max_y; y++, all_r += max_r)
		{
			result[x][y] = all_r;
			for (r = 0; r < max_r; r++, all_c += max_c)
			{
				result[x][y][r] = all_c;
			}
		}
	}

	return result;
}

/* create uInt8 [x][y][r][c] */

uInt8 ****create_4D_uInt8(int max_x, int max_y, int max_r, int max_c)
{
	uInt8 ****all_x = (uInt8 ****) MY_MALLOC( max_x * sizeof (*all_x) );
	uInt8 ***all_y = (uInt8 ***) MY_MALLOC( max_x * max_y * sizeof( *all_y) );
	uInt8 **all_r =
			(uInt8 **) MY_MALLOC( max_x * max_y * max_r * sizeof( *all_r ));
	uInt8 *all_c =
			(uInt8 *) MY_MALLOC( max_x * max_y * max_r * max_c * sizeof(*all_c) );
	uInt8 ****result = all_x;
	int x, y, r;

	for (x = 0; x < max_x; x++, all_y += max_y)
	{
		result[x] = all_y;
		for (y = 0; y < max_y; y++, all_r += max_r)
		{
			result[x][y] = all_r;
			for (r = 0; r < max_r; r++, all_c += max_c)
			{
				result[x][y][r] = all_c;
			}
		}
	}

	return result;
}

/*****************************
 *
 */

//uInt llhread(char * fname, uInt whichframe, nova_str *nova, uInt8 ****llhData) {
uInt8 **** llhread(char * fname, uInt whichframe, nova_str *nova)
{

	FILE* file;
	uInt8 ****llhData;
	uint32_t nosarescan, numBytes;
	/** There are 57 frames worth of data with header (60 bytes).
	 Each frame consists of 5041 (71x71) collimator hole images.
	 Each collimator hole images has 80x160 elements.  The data is arrange to raster down the 80 direction first then the 160 direction.
	 Each element is an 8-bit wide integer
	 */

	file = fopen(fname, "rb");
	if (!file)
	{
		printf("Can not open file %s", fname);
		exit(1);
	}

	/** Read Header */
	//fread(header,HEADER_BYTES,1,file);
	fread((nova_str*) nova, 1, sizeof(nova_str), file);

#if VERBOSE
	printf("hilf        = %s\n", (*nova).hilf);
	printf("noimages    = %u\n", (*nova).noimages);
	printf("detsizex    = %u\n", (*nova).detsizex);
	printf("detsizey    = %u\n", (*nova).detsizey);
	printf("row         = %u\n", (*nova).row);
	printf("col         = %u\n", (*nova).col);
	//printf("nofoc       = %u\n", (*nova).nofoc);
	printf("noframe     = %u\n", (*nova).noframe);
#endif

	nosarescan = nova->noimages / (nova->row * nova->col * nova->noframe);
	if (nosarescan == 1)
	{

	    float framesize;
	    framesize=(nova->row)*(nova->col)*(nova->detsizex)*(nova->detsizey);
	    fseek(file,(whichframe-1)*framesize,SEEK_CUR);
		llhData = (uInt8****) create_4D_uInt8(nova->detsizex, nova->detsizey,
				nova->row, nova->col);
		/** Loop through frames
		 *
		 */

		uint32_t f,  r,c,i, j;

		for (f = 0; f < nova->noframe; f++)
		{

			/** loop through collimator holes
			 *
			 */
			//for ( h = 0; h < (*nova).noimages; h++) {
			// Matlab files are column-major so the order of the loop

				// variables is reversed from usual C standards.
				for (r = 0; r < nova->row; r++)

				{
					for (c = 0; c < nova->col; c++)
								{

#if VERBOSE
					printf("j,k=%i,%i\n\n", j, k);
#endif

					for (j = 0; j < nova->detsizey; j++)
					{


						for (i = 0; i < nova->detsizex; i++)
												{

							if (feof(file))
							{
								printf(
										"premature end of file reached...now exiting\n");
								exit(1);
							}
							//numBytes += fread(&llhData[x][y][j][k],
							numBytes += fread(&llhData[i][j][r][c],
									sizeof(uInt8), 1, file);
#if ALL_ONES
							llhData[i][j][r][c]=1;
#endif

#if VERBOSE
							printf("%x ", llhData[i][j][r][c]);
#endif
						}
#if VERBOSE
						printf("\n");
#endif
					}

#if VERBOSE
				printf("\n");
#endif
				}
			}
		}
	}
	else
	{
		printf("nosarescan = %i, this is not impmented yet...now exiting.\n",
				nosarescan);
		exit(1);
	}

	fclose(file);
	printf("numBytes=%i\n",numBytes);
	return llhData;
}

int directoryExistsQ(const char * path) {

  if (path == NULL) {
    return 0;
  } else {
    DIR * dir;
    if ((dir = opendir(path)) != NULL) {
      closedir(dir);
    }
    return dir != NULL;
  }
}

char * directoryName(const char * path) {
  char * path0 = strdup(path);
  return dirname(path0);
}

void createDirectory(const char * path) {
  if (!directoryExistsQ(path)) {
    mkdir(path, 0777);
  }
}


