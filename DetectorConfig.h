
#ifndef TRT_CUDA_SBDX_DETECTORCONFIG_H_
#define TRT_CUDA_SBDX_DETECTORCONFIG_H_

struct size {
	uint32_t x;
	uint32_t y;
};
struct detector{
	uint32_t sizedet[2];
	float pxlSize;
	float **x_det;
	float **y_det;
	float **z_det;
};

int DetectorConfig(struct detector * d, char *xFile, char *yFile, char *zFile);
int readDetBin(char* fname, uint8_t direction, struct detector *d);
int readDetBin2D(char* fname, uint8_t direction, struct detector *d);



#endif /* DETECTORCONFIG_H_ */
