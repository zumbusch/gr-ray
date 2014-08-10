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

#ifndef KERN_H
#define KERN_H

// ----------------------------------------------------------------------

__global__ void clean () {
  // map from threadIdx/BlockIdx to pixel position
  int i = threadIdx.x + blockIdx.x * BLKX;
  int j = threadIdx.y + blockIdx.y * BLKY;
  int offset = 4 * (i + j * cam.gx);
  cam.ptr[offset + 0] = 0;
  cam.ptr[offset + 1] = 0;
  cam.ptr[offset + 2] = 0;
  cam.ptr[offset + 3] = 255;
};

// ----------------------------------------------------------------------

__device__ void redshift (float r, int c[3]) {
  float co[8] = {0, 0, c[0], c[1], c[2], 0, 0, 0};
  if (r>0.f) {
    r = 4.f * logf (r); // estimate 
    if (r<-2.f) r = -2.f;
    if (r> 2.f) r =  2.f;
  } else
    r = -2.f;
  int s = (int)floorf (r);
  float x = r - s;
  float x1 = 1.f - x;
  for (int i=0; i<3; i++)
    c[i] = (int) (x1*co[s+2+i] + x*co[s+3+i]);
}

__device__ int Plane::isCollision (float x0[8], float xs[8]) {
  float s0 = d, s = d;
  for (int i=1; i<4; i++) {
    s0 -= x0[i]*n[i];
    s -= xs[i]*n[i];
  }
  return (s*s0 <= 0.f);
}

__device__ float Plane::collision (float x0[8], float xs[8]) {
  float s0 = d, s = d;
  for (int i=1; i<4; i++) {
    s0 -= x0[i]*n[i];
    s -= xs[i]*n[i];
  }
  if (s*s0 > 0.f)
    return 2.f; // no collision
  return s0/ (s0-s);
}

__device__ void Plane::color (float x0[8], float xs[8], float g00, int cf[3]) {
  float s0 = d, s = d;
  for (int i=1; i<4; i++) {
    s0 -= x0[i]*n[i];
    s -= xs[i]*n[i];
  }
  s = s0 / (s0-s);
  float x[4];
  for (int i=0; i<4; i++)
    x[i] = x0[i] + (xs[i]-x0[i]) * s;

  int co = 0;
  for (int i=1; i<4; i++)
    co = co ^ (1 & (int)ceilf (x[i]*p));
  float f = 1 - .25f * co;
  for (int i=0; i<3; i++)
    cf[i] = f * col[i];
  float g[4][4];
  metric (x, g);
  redshift (g00/g[0][0], cf);
}

__device__ int Sphere::isCollision (float x0[8], float xs[8]) {
  float a = 0, b = 0, c = -sqr (r);
  for (int k=1; k<4; k++) {
    a += sqr (xs[k]-x0[k]);
    b += (x0[k]-pos[k]) * (xs[k]-x0[k]);
    c += sqr (x0[k]-pos[k]);
  }
  float d = b*b-a*c;
  if (d >= 0.f) {
    float l = - (b+sqrtf (d));
    if ((l >= 0.f) && (l <= a))
      return -1.f;
  }
  return 0.f;
}

__device__ float Sphere::collision (float x0[8], float xs[8]) {
  float a = 0, b = 0, c = -sqr (r);
  for (int k=1; k<4; k++) {
    a += sqr (xs[k]-x0[k]);
    b += (x0[k]-pos[k]) * (xs[k]-x0[k]);
    c += sqr (x0[k]-pos[k]);
  }
  float d = b*b-a*c;
  if (d >= 0.f) {
    float l = - (b+sqrtf (d));
    if ((l >= 0.f) && (l <= a)) {
      return l;
    }
  }
  return 2.f; // no collision
}

__device__ void Sphere::color (float x0[8], float xs[8], float g00, int cf[3]) {
  float a0 = 0.f, b0 = 0.f, c0 = -sqr (r);
  for (int k=1; k<4; k++) {
    a0 += sqr (xs[k]-x0[k]);
    b0 += (x0[k]-pos[k]) * (xs[k]-x0[k]);
    c0 += sqr (x0[k]-pos[k]);
  }
  float d0 = sqr (b0) - a0*c0;
  float l = - (b0+sqrtf (d0)) / a0;
  float x[4];
  for (int k=0; k<4; k++) 
    x[k] = x0[k] + l * (xs[k]-x0[k]) - pos[k];
  float rs = sqrtf (sqr (x[1]) + sqr (x[2]));
  float omega = a / (2.5f * m * sqr (r)); // spinning ball

#ifdef TEX
  // texture
  float p1 = .5f+ (atan2f (x[2], x[1]) + omega*x[0]) / (2.f * M_PIF);
  float p2 = .5f + atan2f (x[3], rs) / M_PIF;
  float u = tex2D (texPlanet, -p1, p2);
  for (int i=0; i<3; i++)
    cf[i] = u * col[i];
#else // TEX
  // checker board
  int p1 = 1 & (int)ceilf (16 * (1+(atan2f (x[2], x[1])+omega*x[0]) / (2*M_PIF)));
  int p2 = 1 & (int)ceilf (16 * (1+atan2f (x[3], rs) / (2*M_PIF)));
  float f = 1.f - .25f * (p1 ^ p2);
  for (int i=0; i<3; i++)
    cf[i] = f * col[i];
#endif // TEX

  float g[4][4];
  for (int i=0; i<4; i++)
    x[i] += pos[i];
  metric (x, g);
  redshift (g00/g[0][0], cf);
}

__device__ float norm4 (float x[4]) {
  float n = 0;
  for (int k=0; k<4; k++)
    n += sqr (x[k]);
  return sqrtf (n);
}

__device__ void func (float xs[8], float xdot[8]) {
  float x[4], v[4];
  for (int k=0; k<4; k++)
    x[k] = xs[k]; // x(t)
  for (int k=0; k<4; k++) {
    v[k] = xs[k+4]; // x'(t)
    xdot[k] = v[k];
  }
  // x^i(t) ' = v^i(t)
  // v^i(t) ' = \sum_{jk} \Gamma^i_{jk}(x(t)) v^j(t) v^k(t)
  // \Gamma^i_{jk} = .5 \sum_{l} g^{il}(d_j g_{kl}+d_k g_{jl}-d_l g_{jk})
  float g[4][4];
  metric (x, g);  // metric
  float gi[4][4];
  for (int i=0; i<4; i++)
    for (int j=0; j<4; j++)
      gi[i][j] = g[i][j];

  // inverse metric
  // float diag[4];
  // factorCholesky_sym2(gi, diag);
  invCholesky_sym2 (gi);

  float h = 2e-3f;  // 1e-3 tune
  float dg[4][4][4]; // metric derivatives
  //    for i=1:4
  //        dg(:,:,i) = metric(x+e(:,i))-metric(x-e(:,i));
  //    end
  //    dg = dg/(2*h); // 2nd order
  for (int k=0; k<4; k++) {
    float g2[4][4];
    float x2[4];
    for (int i=0; i<4; i++)
      x2[i] = x[i];
    x2[k] += h;
    metric (x2, g2);
    for (int i=0; i<4; i++)
      for (int j=0; j<4; j++)
	dg[i][j][k] = (g2[i][j] - g[i][j])/h; // 1st order
  }
  for (int k=0; k<4; k++)
    xdot[k+4] = 0;
  for (int k=0; k<4; k++) {
    float a[4][4];
    for (int i=0; i<4; i++)
      for (int j=0; j<4; j++)
	a[i][j] = .5f * (-dg[k][i][j] + dg[i][j][k] + dg[j][k][i]);

    for (int i=0; i<4; i++)
      for (int j=0; j<4; j++) {
    	float s = 0.f;
    	for (int l=0; l<4; l++)
    	  s += gi[i][l] * a[l][j];
    	//ch[i][j][k] = s; // Christoffel
	xdot[i+4] += v[j] * s * v[k];
      }
  }
}

// ----------------------------------------------------------------------

__global__ void kernel() {
  // map from threadIdx/BlockIdx to pixel position
  int i0 = threadIdx.x & ((1<<VECX)-1);
  int j0 = threadIdx.x >> VECX;
  int i = i0 + (threadIdx.y << VECX) + blockIdx.x * BLKX;
  int j = j0 + (threadIdx.z << VECY) + blockIdx.y * BLKY;
  int offset = 4 * (i + j * cam.gx);

  float x = 2.f * (i-.5f * (cam.gx - 1.f)) / (cam.gx - 1.f);
  float y = 2.f * (j-.5f * (cam.gy - 1.f)) / (cam.gy - 1.f);
  float pd[4];
  for (int k=0; k<4; k++)
    pd[k] = cam.pz[k] + x*cam.px[k] + y*cam.py[k]; // light ray pos+l*pd
  float pdn = 1.f / norm4 (pd);
  for (int k=0; k<4; k++)
    pd[k] *= pdn;
  float g[4][4];
  metric (cam.pos, g);
  float g0 = 0.f;
  for (int k=1; k<4; k++)
    g0 += sqr (cam.pos[k]);
  g0 = sqrtf (g0);
  float a = g[0][0];
  float b = 0.f, c = 0.f;
  for (int k=1; k<4; k++)
    b += 2.f * g[0][k]*pd[k];
  for (int i=1; i<4; i++)
    for (int j=1; j<4; j++)
      c += pd[i] * g[i][j] * pd[j];
  float d = sqr (b)-4*a*c;
  // \sum_{jk} g_{jk}(x(t)) v^j(t) v^k(t) = 0
  pd[0] = - (b+sqrtf (d))/(2*a); // light like, into the past
  float xs[8], xs0[8];
  for (int k=0; k<4; k++)
    xs[k] = cam.pos[k];
  for (int k=0; k<4; k++)
    xs[k+4] = pd[k];

  for (int iter=0; iter<100; iter++) {
    float dt = 2.f;
    for (int k=0; k<SPHERES; k++) {
      float t = .5f * sqrtf (sqr (xs[1]-sph[k].pos[1]) + sqr (xs[2]-sph[k].pos[2]) + sqr (xs[3]-sph[k].pos[3])) /sph[k].r;
      dt = min (dt, t);
    }
    for (int k=0; k<8; k++)
      xs0[k] = xs[k];
    float xdot[8];
    func (xs, xdot);
    // Euler rule
    for (int k=0; k<8; k++)
      xs[k] = xs0[k] + dt * xdot[k];
    /*
    // midpoint rule
    for (int k=0; k<8; k++)
    xs[k] += (.5f * dt) * xdot[k];
    func (xs, xdot);
    for (int k=0; k<8; k++)
    xs[k] = xs0[k] + dt * xdot[k];
    */
    int b = 0;
    for (int k=0; k<PLANES; k++)
      b = b || plane[k].isCollision (xs0, xs);
    for (int k=0; k<SPHERES; k++)
      b = b || sph[k].isCollision (xs0, xs);
    if (b) {
      break;
    }
  }

  int co[3] = {0, 0, 0};
  float n = 2.f;
  for (int k=0; k<SPHERES; k++) {
    float n0 = sph[k].collision (xs0, xs);
    if (n > n0) {
      n = n0;
      sph[k].color (xs0, xs, a, co);
    }
  }
  for (int k=0; k<PLANES; k++) {
    float n0 = plane[k].collision (xs0, xs);
    if (n > n0) {
      n = n0;
      plane[k].color (xs0, xs, a, co);
    }
  }
  for (int k=0; k<3; k++)
    cam.ptr[offset + k] = co[k];
}

// ----------------------------------------------------------------------

#endif // KERN_H