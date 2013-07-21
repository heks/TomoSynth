
#ifndef WRITETIF_H_
#define WRITETIF_H_

#include <stdlib.h>
#include "stdHdr.h"

typedef FloatSP IP_PIXEL;
typedef IP_PIXEL* IP_PIMAGE;

extern void writeTif(const char* fname,IP_PIMAGE pDstImage,SNativeInt nDstRows,SNativeInt nDstColumns);

#endif /* WRITETIF_H_ */
