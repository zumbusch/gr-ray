// *-*-C++-*-*

// ----------------------------------------------------------------------
//
// Copyright (c) 2011, 2014, Gerhard Zumbusch.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//     * The names of its contributors may not be used to endorse or
//       promote products derived from this software without specific
//       prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

// This software contains source code provided by NVIDIA Corporation.
//
// ----------------------------------------------------------------------

/*
  relativistic raytracer:
  spinning spheres: Kerr metric in Kerr-Schild coordinates
  walls
  red-shift
*/


#include "ppm.h"
#include "cpu_anim.h"
#include <unistd.h>

#define DIMX 512
#define DIMY 512

// warp size 32 = 2^5 =  2^(VECX+VECY)
#define VECX 3
#define VECY 2
// multithreading =  thread.y, thread.z
#define THREADX 2
#define THREADY 2
// 
#define BLKX (THREADX<<VECX)
#define BLKY (THREADY<<VECY)

#define TEX
// #define PROFILE 10

#ifndef sqr
#define sqr(x) ((x)*(x))
#endif
#define M_PIF (3.141592653589793f)

#ifdef TEX
texture< float, 2, cudaReadModeElementType >  texPlanet;
#endif
cudaArray *arrayPlanet;

struct Sphere {
  float pos[4]; // center
  float m, r, a; // mass, radius, spin, r>2*m
  float col[3];
  // texture 
  __device__ int isCollision (float x0[8], float x[8]);
  __device__ float collision (float x0[8], float x[8]);
  __device__ void color (float x0[8], float xs[8], float g00, int cf[3]);
};

#define SPHERES 2

__constant__ Sphere sph[SPHERES>0 ? SPHERES : 1];
Sphere sph_host[SPHERES];

#include "metric.h"


struct Camera { // camera position and direction
  float pos[4], px[4], py[4], pz[4];
  int gx, gy; // screen resolution
  unsigned char *ptr;
};

__constant__ Camera cam;
Camera cam_host;

struct Plane {
  float n[4]; // normal vector
  float d; // distance
  float p; // pattern size
  float col[3]; 
  __device__ int isCollision (float x0[8], float x[8]);
  __device__ float collision (float x0[8], float x[8]);
  __device__ void color (float x0[8], float xs[8], float g00, int cf[3]);
};

#define PLANES 5

__constant__ Plane plane[PLANES];


#include <cuda.h>
#include "cholesky.h"
#include "kern.h"

// globals needed by the update routine
struct DataBlock {
  unsigned char *dev_bitmap;
  cudaEvent_t start, stop;
  float totalTime, frames;
  int evolve;
  CPUAnimBitmap *bitmap;
};

void anim_reshape (DataBlock *d, int x, int y) {
  // sync cuda
  HANDLE_ERROR (cudaMemcpy (d->bitmap->get_ptr (), d->dev_bitmap,
			    64, cudaMemcpyDeviceToHost));
  // resize memory
  if (d->bitmap->pixels)
    delete[] d->bitmap->pixels;
  d->bitmap->pixels = new unsigned char[d->bitmap->size];
  HANDLE_ERROR (cudaFree (d->dev_bitmap));
  HANDLE_ERROR (cudaMalloc ((void**)&d->dev_bitmap,
   			    d->bitmap->size));
  cam_host.px[2] = x/ (float)y;
  cam_host.gx = max (2,x);
  cam_host.gy = max (2,y);
  cam_host.ptr = d->dev_bitmap;
  // dim3 grids (cam_host.gx/BLKX, cam_host.gy/BLKY);
  // dim3 threads (BLKX, BLKY);
  // clean <<<grids,threads>>> ();
  // HANDLE_LAST_ERROR("Kernel execution failed");
}

void anim_clickdrag (DataBlock *d, float rx, float ry, float tx, float ty, float tz) {
  cam_host.pos[1] = -20+5*tz;
  cam_host.pos[2] = -5*tx;
  cam_host.pos[3] = -5*ty;
  cam_host.pz[2] = -.3*rx;
  cam_host.pz[3] = -.3*ry;
}

void anim_gpu (DataBlock *d) {
  if (d->evolve || d->bitmap->update) {
    d->bitmap->update = 0;
    HANDLE_ERROR (cudaMemcpyToSymbol (cam, &cam_host, sizeof (Camera)));
    HANDLE_ERROR (cudaEventRecord (d->start, 0));

    // generate a bitmap
    dim3 grids (cam_host.gx/BLKX, cam_host.gy/BLKY, 1);
    dim3 threads (1<<(VECX+VECY), THREADX, THREADY);
    kernel <<<grids, threads>>> ();
    HANDLE_LAST_ERROR ("Kernel execution failed");
    // copy our bitmap back from the GPU for display
    HANDLE_ERROR (cudaMemcpy (d->bitmap->get_ptr (), d->dev_bitmap,
			      d->bitmap->size,
			      cudaMemcpyDeviceToHost));

    HANDLE_ERROR (cudaEventRecord (d->stop, 0));
    HANDLE_ERROR (cudaEventSynchronize (d->stop));
    float elapsedTime;
    HANDLE_ERROR (cudaEventElapsedTime (&elapsedTime,
					d->start, d->stop));
    d->totalTime += elapsedTime;
    ++d->frames;
#ifdef PROFILE
    if (d->frames > PROFILE) exit(0);
#endif // PROFILE
    printf ("time :  %3.1f ms\n", elapsedTime);
    if (d->evolve)
      cam_host.pos[0] += .5; // time evolution
  } else
    usleep (20000);
}

// clean up memory allocated on the GPU
void anim_exit (DataBlock *d) {
#ifdef TEX
  cudaUnbindTexture (texPlanet);
#endif
  HANDLE_ERROR (cudaFree (d->dev_bitmap));
  HANDLE_ERROR (cudaFreeArray (arrayPlanet));
  HANDLE_ERROR (cudaEventDestroy (d->start));
  HANDLE_ERROR (cudaEventDestroy (d->stop));
}

void anim_key (DataBlock *d, unsigned char k) {
  if (k == char ('e')) {
    d->evolve = 1 - d->evolve;
    printf ("evolve\n");
  }
#if SPHERES>0
  bool change = false;
  if (k == char ('N') || k == char ('n')) {
    change = true;
    float f = k == char ('N') ? 1/.9 : .9;
    sph_host[1].m *= f;
    sph_host[1].r *= f;
    printf ("mass %g %g\n", sph_host[0].m, sph_host[1].m);
  }
  if (k == char ('M') || k == char ('m')) {
    change = true;
    float f = k == char ('M') ? 1/.9 : .9;
    sph_host[0].m *= f;
    sph_host[0].r *= f;
    printf ("mass %g %g\n", sph_host[0].m, sph_host[1].m);
  }
  if (k == char ('O') || k == char ('o')) {
    change = true;
    float p = k == char ('O') ? .1 : -.1;
    sph_host[1].a += p;
    printf ("spin %g %g\n", sph_host[0].a, sph_host[1].a);
  }
  if (k == char ('P') || k == char ('p')) {
    change = true;
    float p = k == char ('P') ? .1 : -.1;
    sph_host[0].a += p;
    printf ("spin %g %g\n", sph_host[0].a, sph_host[1].a);
  }
  if (change) {
    HANDLE_ERROR (cudaMemcpyToSymbol (sph, sph_host, 
				      sizeof (Sphere) * SPHERES));
  }
#endif // SPHERES
}

void start_tex () {
#ifdef TEX
  float *d = 0; 
  unsigned int x, y, c, s;
  loadPPM ("2048px-Equirectangular-projection.pgm", &d, &x, &y, &c);
  // printf ("%d %d %d\n",x,y,c);
  s = x * y * c * sizeof(float);
  // Allocate CUDA array in device memory
  cudaChannelFormatDesc channelDesc =
    cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  HANDLE_ERROR (cudaMallocArray (&arrayPlanet, &channelDesc, x, y));
  HANDLE_ERROR (cudaMemcpyToArray (arrayPlanet, 0, 0, d, s,
				   cudaMemcpyHostToDevice));
  free (d);
  // Set texture parameters
  texPlanet.addressMode[0] = cudaAddressModeWrap;
  texPlanet.addressMode[1] = cudaAddressModeWrap;
  texPlanet.filterMode = cudaFilterModeLinear;
  texPlanet.normalized = true;
  // Bind the array to the texture reference
  HANDLE_LAST_ERROR ("execution failed");
  HANDLE_ERROR (cudaBindTextureToArray (texPlanet, arrayPlanet, channelDesc));
#endif // TEX
}

int main (int argc, char **argv)
{
  int devID = 0;
  HANDLE_ERROR (cudaSetDevice (devID));
  cudaDeviceProp deviceProp;
  HANDLE_ERROR (cudaGetDeviceProperties (&deviceProp, devID));
  printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
  start_tex ();

  DataBlock data;
  // capture the start time
  cudaEvent_t start, stop;
  HANDLE_ERROR (cudaEventCreate (&start));
  HANDLE_ERROR (cudaEventCreate (&stop));
  HANDLE_ERROR (cudaEventRecord (start, 0));

  CPUAnimBitmap bitmap (DIMX, DIMY, &data);
  data.bitmap = &bitmap;
  data.totalTime = 0;
  data.frames = 0;
  data.evolve = 0;
#ifdef PROFILE
  data.evolve = 1;
#endif

  // allocate memory on the GPU for the output bitmap
  HANDLE_ERROR (cudaMalloc ((void**)&data.dev_bitmap,
			    bitmap.size));

#if SPHERES>0
  sph_host[0] = (Sphere){{0, 0, 0, 0},
			 .5, 2., .6,
			 {160, 200, 255}};
#if SPHERES>1
  sph_host[1] = (Sphere){{0, 0, -3, -3},
			 .125, .5, .3,
			 {150, 150, 150}};
#endif
  HANDLE_ERROR (cudaMemcpyToSymbol (sph, sph_host, 
				    sizeof (Sphere) * SPHERES));
#endif

  Plane *temp_p = (Plane*)malloc (sizeof (Plane) * PLANES);
  float r = 7, r2 = 6 / (2*r); // box size, nr. of squares
  temp_p[0] = (Plane){{0, 1, 0, 0},
		      r, r2,
		      {50, 255, 50}};
#if PLANES>1
  temp_p[1] = (Plane){{0, 0, 1, 0},
		      r, r2,
		      {250, 255, 50}};
  temp_p[2] = (Plane){{0, 0, -1, 0},
		      r, r2,
		      {250, 255, 50}};
  temp_p[3] = (Plane){{0, 0, 0, 1},
		      r, r2,
		      {50, 255, 250}};
  temp_p[4] = (Plane){{0, 0, 0, -1},
		      r, r2,
		      {50, 255, 250}};
#endif
  HANDLE_ERROR (cudaMemcpyToSymbol (plane, temp_p, 
				    sizeof (Plane) * PLANES));
  free (temp_p);

  cam_host = (Camera){ {0, -20, 0, 0},
		       {0, 0, DIMX/ (float)DIMY, 0},
		       {0, 0, 0, 1},
		       {0, 2, 0, 0},
		       max (2, DIMX), max (2, DIMY), data.dev_bitmap };
  HANDLE_ERROR (cudaMemcpyToSymbol (cam, &cam_host, sizeof (Camera)));
  HANDLE_ERROR (cudaEventCreate (&data.start));
  HANDLE_ERROR (cudaEventCreate (&data.stop));

  printf ("move camera [mouse left/right]\nrotate camera [shift+mouse left or mouse middle]\ntoggle evolve [e]\nchange mass [mMnN], spin [oOpP]\nexit [esc], full screen [F5]\n");
  // display
  bitmap.click_drag ((void (*) (void*,float,float,float,float,float)) anim_clickdrag,
  		     (void (*) (void*,int,int)) anim_reshape);
  bitmap.anim_and_exit ((void (*) (void*))anim_gpu,
  			(void (*) (void*))anim_exit,
  			(void (*) (void*,unsigned char))anim_key);
}
