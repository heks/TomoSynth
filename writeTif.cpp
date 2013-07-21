
#include <stdio.h>
#include <tiffio.h>
#include <stdint.h>
#include "writeTif.h"

void writeTif(const char* fname , IP_PIMAGE pDstImage , SNativeInt nDstRows ,
        SNativeInt nDstColumns)
{

    int r, c;
    uint16_t *output;

    output = (uint16_t *) malloc(sizeof(uint16_t) * nDstRows * nDstColumns);

    TIFF *image;

    // Open the TIFF file
    if ((image = TIFFOpen(fname , "w")) == NULL)
    {
      printf("Could not open %s for writing\n",fname);
        exit(42);
    }

    // We need to set some values for basic tags before we can add any data
    TIFFSetField(image , TIFFTAG_IMAGEWIDTH , nDstColumns);
    TIFFSetField(image , TIFFTAG_IMAGELENGTH , nDstRows);
    TIFFSetField(image , TIFFTAG_BITSPERSAMPLE , 16);
    TIFFSetField(image , TIFFTAG_SAMPLESPERPIXEL , 1);
    TIFFSetField(image , TIFFTAG_ROWSPERSTRIP , nDstRows);
    //TIFFSetField(image , TIFFTAG_SAMPLEFORMAT , SAMPLEFORMAT_UINT);

    //TIFFSetField(image, TIFFTAG_COMPRESSION, COMPRESSION_CCITTFAX4);
    TIFFSetField(image , TIFFTAG_PHOTOMETRIC , PHOTOMETRIC_MINISBLACK);
    TIFFSetField(image, TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);
    TIFFSetField(image , TIFFTAG_PLANARCONFIG , PLANARCONFIG_CONTIG);

    //TIFFSetField(image, TIFFTAG_XRESOLUTION, 150.0);
    //TIFFSetField(image, TIFFTAG_YRESOLUTION, 150.0);
    //TIFFSetField(image, TIFFTAG_RESOLUTIONUNIT, RESUNIT_INCH);

    // Write the information to the file

    for(r = 0; r < nDstRows; r++)
    {

        for(c = 0; c < nDstColumns; c++)
        {

            output[r * nDstColumns + c]= (uint16_t) floor(pDstImage[r * nDstColumns + c]);

        }
    }

    if (TIFFWriteRawStrip(image , 0 , output , nDstColumns * nDstRows*sizeof(uint16_t)) == 0)
    {
        printf("Error  writing output.tif \n");
        exit(43);
    }


/*   if(TIFFWriteEncodedStrip(image, 0, output, nDstRows*nDstColumns)==0){
 printf("Error  writing output.tif \n");
 exit(43);
 }*/

// Close the file
TIFFClose (image);

free (output);

}
