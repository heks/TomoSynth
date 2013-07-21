
#include <stdint.h>
#include <stdio.h>

#include "DetectorConfig.h"
#include "trt_utils.h"

int readDetBin2D(char* fname, uint8_t direction, struct detector *d)
{
	FILE* file;
	file = fopen(fname, "rb");
	if (!file) {
		printf("    Can not open file %s\n",fname);
		exit(1);
	}

	// Read the dimensions
	fread((uint32_t *) &d->sizedet[0], 1, sizeof(uint32_t), file);
	fread((uint32_t *) &d->sizedet[1], 1, sizeof(uint32_t), file);
	
	uint32_t numBytes;

	if(direction==0) // x-direction
	{
		// Xfile is the only one with pxlsize
		numBytes=fread((float *) &d->pxlSize, 1, sizeof(float), file);
		printf("    pxlSize=%f \n",d->pxlSize);
	}
	//allocate space for the direction grid

	float** temp_det;
	//temp_det = (float *) malloc(sizeof(float)* (d->sizedet[0])* (d->sizedet[1]));
	temp_det=create_2D_float(d->sizedet[0],d->sizedet[1]);
	// Matlab is column-major so we read element-by-element and stuff into array;


	for( int j=0; j< d->sizedet[1]; j++){
		for(int i=0; i< d->sizedet[0] ; i++){
			fread((float *) &temp_det[i][j] , 1,sizeof(float) , file);
#if VERBOSE
			            printf("%f ",temp_det[i][j]);
#endif
		}
#if VERBOSE
		printf("\n");
#endif
	}
#if VERBOSE
	printf("\n");
	for(int i=0; i< d->sizedet[0] ; i++){
		for( int j=0; j< d->sizedet[1]; j++){
			printf("%f ",temp_det[i][j]);
		}
		printf("\n");
	}
	printf("\n");
#endif
	// Read the direction grid


	if(direction==0) // x-direction
	{
		printf("    x_det\n");
		d->x_det =temp_det;
	}
	else if(direction==1 ) //y-direction
	{
		printf("    y_det\n");
		d->y_det=temp_det;
	}
	else if(direction==2 ) // z-direction
	{
		printf("    z_det\n");
		d->z_det=temp_det;
	}
	else {
		printf("failed to determine which detector direction to fill..now exiting\n");
		exit(1);
	}


	fclose(file);

	//free(temp_det); Don't free here, the pointer has been re-assigned.
	// Free d->x_det, d->y_det, d->z_det during final cleanup.

	return numBytes;
}



int DetectorConfig(struct detector *d, char *xFile, char *yFile, char *zFile){

	//struct size sizedet;
	readDetBin2D((char *)xFile, 0, d);
	readDetBin2D((char *)yFile, 1, d);
	readDetBin2D((char *)zFile, 2, d);

	//printf("    sizedet.x=%u, sizedet.y=%u\n",d->sizedet[0], d->sizedet[1]);
	printf("    pxlSize=%f \n",d->pxlSize);

	return 0;

}


