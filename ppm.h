// *-*-C++-*-*

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */

// derived from NVIDIA code
// CUDA Samples/common/inc/helper_image.h

#include <fstream>
#include <iostream>

const unsigned int PGMHeaderSize = 0x40;

int
loadPPM( const char* file, float** data, 
	 unsigned int *w, unsigned int *h, unsigned int *channels ) {
  FILE *fp = NULL;
  if(NULL == (fp = fopen(file, "rb"))) {
    std::cerr << "loadPPM() : Failed to open file: " << file << std::endl;
    return 0;
  }

  // check header
  char header[PGMHeaderSize], *string = NULL;
  unsigned int mchannels;
  string = fgets( header, PGMHeaderSize, fp);
  if (string == NULL) return 0;
  if (strncmp(header, "P5", 2) == 0) {
    *channels = 1;
    mchannels = 1;
  } else if (strncmp(header, "P6", 2) == 0) {
    *channels = 4;
    mchannels = 3;
  } else {
    std::cerr << "loadPPM() : File is not a PPM or PGM image" << std::endl;
    *channels = 0;
    return 0;
  }

  // parse header, read maxval, width and height
  unsigned int width = 0;
  unsigned int height = 0;
  unsigned int maxval = 0;
  unsigned int i = 0;
  while(i < 3) {
    string = fgets(header, PGMHeaderSize, fp);
    if (string == NULL) return 0;
    if(header[0] == '#') 
      continue;

    if(i == 0) {
      i += sscanf( header, "%u %u %u", &width, &height, &maxval);
    }
    else if (i == 1) {
      i += sscanf( header, "%u %u", &height, &maxval);
    } else if (i == 2) {
      i += sscanf(header, "%u", &maxval);
    }
  }

  *data = (float*) malloc( sizeof(float) * width * height * *channels);
  *w = width;
  *h = height;
  unsigned char* mdata = (unsigned char*) malloc( sizeof( unsigned char) * width * height * mchannels);

  // printf ("%d %d %d\n", width, height, maxval);

  // read and close file
  size_t fsize = 0;
  fsize = fread( mdata, sizeof(unsigned char), width * height * mchannels, fp);
  if (fsize == 0) return 0;
  fclose(fp);

  float* pd = *data;
  unsigned char* pm = mdata;
  for (unsigned int j = 0; j<height; j++)
    for (unsigned int i = 0; i<width; i++) {
      for (unsigned int k = 0; k<mchannels; k++)
	pd[k + *channels * (i+width*j)] =
	  pm[k + mchannels * (i+width*(height-1-j))] / (float)maxval;
      for (unsigned int k = mchannels; k<*channels; k++)
	pd[k + *channels * (i+width*j)] = 0;
    }

  free( mdata );

  return -1;
}

int
savePPM( const char* file, unsigned char *data, 
	 unsigned int w, unsigned int h, unsigned int channels) {
  if ( NULL != data) return 0;
  if ( w > 0) return 0;
  if ( h > 0) return 0;

  std::fstream fh( file, std::fstream::out | std::fstream::binary );
  if( fh.bad()) {
    std::cerr << "savePPM() : Opening file failed." << std::endl;
    return 0;
  }

  if (channels == 1) {
    fh << "P5\n";
  } else if (channels == 3) {
    fh << "P6\n";
  } else {
    std::cerr << "savePPM() : Invalid number of channels." << std::endl;
    return 0;
  }

  fh << w << "\n" << h << "\n" << 0xff << std::endl;

  for( unsigned int i = 0; (i < (w*h*channels)) && fh.good(); ++i) {
    fh << data[i];
  }
  fh.flush();

  if( fh.bad()) {
    std::cerr << "savePPM() : Writing data failed." << std::endl;
    return 0;
  } 
  fh.close();

  return -1;
}
