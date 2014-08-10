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
// ----------------------------------------------------------------------

#ifndef METRIC_H
#define METRIC_H

// flat space, Minkowski metric
__device__ void metric_flat (float x[4], float g[4][4]) {
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      g[i][j] = 0.f;
  g[0][0] = 1.f;
  for (int i=1; i<4; i++)
    g[i][i] = -1.f;
}

// linear wave
__device__ void metric_wave (float x[4], float g[4][4]) {
  float r = .1f, o = 2., k[4] = {-1.f, -1.f, 0.f, 0.f}, y = 0.f;
  for (int i=0; i<4; i++)
    y += k[i] * x[i]; 
  float s = r * sinf (o * y);
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      g[i][j] = s;//0;
  g[0][0] = 1.f + s;
  for (int i=1; i<4; i++)
    g[i][i] = -1.f + s;
}

// sum of Schwarzschild metrics in harmonic coord
__device__ void metric_bh (float xx[4], float g[4][4]) {
  for (int i=0; i<4; ++i)
    for (int j=0; j<4; ++j)
      g[i][j] = 0.f;
  for (int k=0; k<SPHERES; k++) {
    float x[4];
    for (int i=0; i<4; ++i)
      x[i] = xx[i] - sph[k].pos[i];
    float r = sqrtf (sqr (x[1])+sqr (x[2])+sqr (x[3]));
    float mr = sph[k].m/r;
    g[0][0] += (1.f - mr) / (1.f + mr);
    float f = -sqr (mr/r) * (1+mr) / (1-mr);
    float d = sqr (1.f + mr);
    for (int i=1; i<4; ++i) {
      for (int j=1; j<4; ++j)
	g[i][j] += f * x[i] * x[j];
      g[i][i] -= d;
    }
  }
}

// sum of Kerr metrics in Kerr-Schild coord
__device__ void metric_ks (float xx[4], float g[4][4]) {
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      g[i][j] = 0.f;
  g[0][0] = 1.f;
  for (int i=1; i<4; i++)
    g[i][i] = -1.f;
  for (int k=0; k<SPHERES; k++) {
    float x[4];
    for (int i=0; i<4; ++i)
      x[i] = xx[i] - sph[k].pos[i];
    float a = sph[k].a; 
    float m = sph[k].m; 
    float a2 = sqr (a);
    float r02 = sqr (x[1]) + sqr (x[2]) + sqr (x[3]);
    float b = .5f * (r02 - a2);
    float r2 = b + sqrtf (sqr (b) + a2 * sqr (x[3]));
    float r = sqrtf (r2);
    float r2a2 = r2 + a2;
    float y[4];
    y[0] = 1;
    y[1] = (r*x[1] + a*x[2]) / r2a2;
    y[2] = (r*x[2] - a*x[1]) / r2a2;
    y[3] = x[3] / r;
    float f = r2 / (sqr (r2) + a2 * sqr (x[3])) * 2.f * m * r;
    for (int i=0; i<4; i++)
      for (int j=0; j<4; j++)
	g[i][j] -= f * y[i] * y[j];
  }
}

__device__ void metric (float x[4], float g[4][4]) {
  return metric_ks (x, g);
}


#endif // METRIC_H
