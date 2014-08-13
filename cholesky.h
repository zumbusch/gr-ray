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

#ifndef CHOLESKY_H
#define CHOLESKY_H

#ifndef sqr
#define sqr(x) ((x)*(x))
#endif

typedef float real;

// ----------------------------------------------------------------------
// solve equation system
// symmetric matrix A

__device__ inline void factorSubstCholesky(float a[4][4], float b[4]) {
  // LDLt Cholesky decomposition
  // L in c
  // D in c
  float c11 = a[0][0];
  float c21 = a[1][0];
  float c22 = a[1][1];
  float c31 = a[2][0];
  float c32 = a[2][1];
  float c33 = a[2][2];
  float c41 = a[3][0];
  float c42 = a[3][1];
  float c43 = a[3][2];
  float c44 = a[3][3];
  c11 = c11;
  c21 = (c21) / c11;
  c31 = (c31) / c11;
  c41 = (c41) / c11;
  c22 = c22 - c11 * (c21 * c21);
  c32 = (c32 - c11 * (c31 * c21)) / c22;
  c42 = (c42 - c11 * (c41 * c21)) / c22;
  c33 = c33 - c11 * (c31 * c31) - c22 * (c32 * c32);
  c43 = (c43 - c11 * (c41 * c31) - c22 * (c42 * c32)) / c33;
  c44 = c44 - c11 * (c41 * c41) - c22 * (c42 * c42) - c33 * (c43 * c43);
  float a1 = b[0];
  float a2 = b[1];
  float a3 = b[2];
  float a4 = b[3];
  a2 = a2 - a1 * c21;
  a3 = a3 - a1 * c31 - a2 * c32;
  a4 = a4 - a1 * c41 - a2 * c42 - a3 * c43;
  a1 = a1 / c11;
  a2 = a2 / c22;
  a3 = a3 / c33;
  a4 = a4 / c44;
  a3 = a3 - a4 * c43;
  a2 = a2 - a3 * c32 - a4 * c42;
  a1 = a1 - a2 * c21 - a3 * c31 - a4 * c41;
  b[0] = a1;
  b[1] = a2;
  b[2] = a3;
  b[3] = a4;
}

__device__ inline void factorSubstCholesky(float a[10], float b[4]) {
  // LDLt Cholesky decomposition
  // L in a
  // D in a
  float a11 = a[0];
  float a21 = a[1];
  float a22 = a[2];
  float a31 = a[3];
  float a32 = a[4];
  float a33 = a[5];
  float a41 = a[6];
  float a42 = a[7];
  float a43 = a[8];
  float a44 = a[9];
  a21 = (a21) / a11;
  a31 = (a31) / a11;
  a41 = (a41) / a11;
  a22 = a22 - a11 * (a21 * a21);
  a32 = (a32 - a11 * (a31 * a21)) / a22;
  a42 = (a42 - a11 * (a41 * a21)) / a22;
  a33 = a33 - a11 * (a31 * a31) - a22 * (a32 * a32);
  a43 = (a43 - a11 * (a41 * a31) - a22 * (a42 * a32)) / a33;
  a44 = a44 - a11 * (a41 * a41) - a22 * (a42 * a42) - a33 * (a43 * a43);
  float b1 = b[0];
  float b2 = b[1];
  float b3 = b[2];
  float b4 = b[3];
  b2 = b2 - b1 * a21;
  b3 = b3 - b1 * a31 - b2 * a32;
  b4 = b4 - b1 * a41 - b2 * a42 - b3 * a43;
  b1 = b1 / a11;
  b2 = b2 / a22;
  b3 = b3 / a33;
  b4 = b4 / a44;
  b3 = b3 - b4 * a43;
  b2 = b2 - b3 * a32 - b4 * a42;
  b1 = b1 - b2 * a21 - b3 * a31 - b4 * a41;
  b[0] = b1;
  b[1] = b2;
  b[2] = b3;
  b[3] = b4;
}

template <int N>
__device__ inline void factorSubstCholesky_0(real a[N][N], real b[N]) {
  // LDLt Cholesky decomposition
  // L in a
  // D in a
  // inv D in id
  real id[N];
  for (int j=0; j<N; j++) {
    real s = a[j][j];
    for (int k=0; k<j; k++) {
      real t = a[j][k];
       s -= a[k][k] * (t * t);
    }
    a[j][j] = s;
    id[j] = 1.f / s;
    for (int i=j+1; i<N; i++) {
      real s = a[i][j];
       for (int k=0; k<j; k++) {
	 s -= a[k][k] * a[i][k] * a[j][k];
       }
       a[i][j] = s * id[j];
    }
  }

  for (int i=1; i<N; i++) {
    for (int k=0; k<i; k++)
      b[i] -= b[k] * a[i][k];
  }
  for (int i=0; i<N; i++)
    b[i] *= id[i];
  for (int i=N-2; i>=0; i--) {
    for (int k=i+1; k<N; k++)
      b[i] -= b[k] * a[k][i];
  }
}


// ----------------------------------------------------------------------
// factorize symmetric matrix A

__device__ inline void factorCholesky(float a[4][4]) {
  // LDLt Cholesky decomposition
  // L in c
  // D in c
  float c11 = a[0][0];
  float c21 = a[1][0];
  float c22 = a[1][1];
  float c31 = a[2][0];
  float c32 = a[2][1];
  float c33 = a[2][2];
  float c41 = a[3][0];
  float c42 = a[3][1];
  float c43 = a[3][2];
  float c44 = a[3][3];
  c11 = c11;
  c21 = (c21) / c11;
  c31 = (c31) / c11;
  c41 = (c41) / c11;
  c22 = c22 - c11 * (c21 * c21);
  c32 = (c32 - c11 * (c31 * c21)) / c22;
  c42 = (c42 - c11 * (c41 * c21)) / c22;
  c33 = c33 - c11 * (c31 * c31) - c22 * (c32 * c32);
  c43 = (c43 - c11 * (c41 * c31) - c22 * (c42 * c32)) / c33;
  c44 = c44 - c11 * (c41 * c41) - c22 * (c42 * c42) - c33 * (c43 * c43);
  a[0][0] = c11;
  a[1][0] = c21;
  a[1][1] = c22;
  a[2][0] = c31;
  a[2][1] = c32;
  a[2][2] = c33;
  a[3][0] = c41;
  a[3][1] = c42;
  a[3][2] = c43;
  a[3][3] = c44;
}

__device__ inline void factorCholesky(float a[10]) {
  // LDLt Cholesky decomposition
  // L in a
  // D in a
  float a11 = a[0];
  float a21 = a[1];
  float a22 = a[2];
  float a31 = a[3];
  float a32 = a[4];
  float a33 = a[5];
  float a41 = a[6];
  float a42 = a[7];
  float a43 = a[8];
  float a44 = a[9];
  a21 = (a21) / a11;
  a31 = (a31) / a11;
  a41 = (a41) / a11;
  a22 = a22 - a11 * (a21 * a21);
  a32 = (a32 - a11 * (a31 * a21)) / a22;
  a42 = (a42 - a11 * (a41 * a21)) / a22;
  a33 = a33 - a11 * (a31 * a31) - a22 * (a32 * a32);
  a43 = (a43 - a11 * (a41 * a31) - a22 * (a42 * a32)) / a33;
  a44 = a44 - a11 * (a41 * a41) - a22 * (a42 * a42) - a33 * (a43 * a43);
  a[0] = a11;
  a[1] = a21;
  a[2] = a22;
  a[3] = a31;
  a[4] = a32;
  a[5] = a33;
  a[6] = a41;
  a[7] = a42;
  a[8] = a43;
  a[9] = a44;
}

template <int N>
__device__ inline void factorCholesky_0(real a[N][N]) {
  // LDLt Cholesky decomposition
  // L in a
  // D in a
  for (int j=0; j<N; j++) {
    real s = a[j][j];
    for (int k=0; k<j; k++) {
      real t = a[j][k];
       s -= a[k][k] * (t * t);
    }
    a[j][j] = s;
    for (int i=j+1; i<N; i++) {
      real s = a[i][j];
       for (int k=0; k<j; k++) {
	 s -= a[k][k] * a[i][k] * a[j][k];
       }
       a[i][j] = s / a[j][j];
    }
  }
}

// ----------------------------------------------------------------------
// subst rhs, factorized symmetric matrix A

__device__ inline void substCholesky(float a[4][4], float b[4]) {
  // LDLt Cholesky decomposition
  // L in c
  // D in c
  float c11 = a[0][0];
  float c21 = a[1][0];
  float c22 = a[1][1];
  float c31 = a[2][0];
  float c32 = a[2][1];
  float c33 = a[2][2];
  float c41 = a[3][0];
  float c42 = a[3][1];
  float c43 = a[3][2];
  float c44 = a[3][3];
  float a1 = b[0];
  float a2 = b[1];
  float a3 = b[2];
  float a4 = b[3];
  a2 = a2 - a1 * c21;
  a3 = a3 - a1 * c31 - a2 * c32;
  a4 = a4 - a1 * c41 - a2 * c42 - a3 * c43;
  a1 = a1 / c11;
  a2 = a2 / c22;
  a3 = a3 / c33;
  a4 = a4 / c44;
  a3 = a3 - a4 * c43;
  a2 = a2 - a3 * c32 - a4 * c42;
  a1 = a1 - a2 * c21 - a3 * c31 - a4 * c41;
  b[0] = a1;
  b[1] = a2;
  b[2] = a3;
  b[3] = a4;
}

__device__ inline void substCholesky(float a[10], float b[4]) {
  // LDLt Cholesky decomposition
  // L in a
  // D in a
  float a11 = a[0];
  float a21 = a[1];
  float a22 = a[2];
  float a31 = a[3];
  float a32 = a[4];
  float a33 = a[5];
  float a41 = a[6];
  float a42 = a[7];
  float a43 = a[8];
  float a44 = a[9];
  float b1 = b[0];
  float b2 = b[1];
  float b3 = b[2];
  float b4 = b[3];
  b2 = b2 - b1 * a21;
  b3 = b3 - b1 * a31 - b2 * a32;
  b4 = b4 - b1 * a41 - b2 * a42 - b3 * a43;
  b1 = b1 / a11;
  b2 = b2 / a22;
  b3 = b3 / a33;
  b4 = b4 / a44;
  b3 = b3 - b4 * a43;
  b2 = b2 - b3 * a32 - b4 * a42;
  b1 = b1 - b2 * a21 - b3 * a31 - b4 * a41;
  b[0] = b1;
  b[1] = b2;
  b[2] = b3;
  b[3] = b4;
}

template <int N>
__device__ inline void substCholesky_0(real a[N][N], real b[N]) {
  // LDLt Cholesky decomposition
  // L in a
  // D in a

  for (int i=1; i<N; i++) {
    for (int k=0; k<i; k++)
      b[i] -= b[k] * a[i][k];
  }
  for (int i=0; i<N; i++)
    b[i] /= a[i][i];
  for (int i=N-2; i>=0; i--) {
    for (int k=i+1; k<N; k++)
      b[i] -= b[k] * a[k][i];
  }
}


// ----------------------------------------------------------------------
// invert symmetric matrix A

__device__ inline void invCholesky(real a[4][4]) {
  float c11 = a[0][0];
  float c21 = a[1][0];
  float c22 = a[1][1];
  float c31 = a[2][0];
  float c32 = a[2][1];
  float c33 = a[2][2];
  float c41 = a[3][0];
  float c42 = a[3][1];
  float c43 = a[3][2];
  float c44 = a[3][3];
  c11 = c11;
  c21 = (c21) / c11;
  c31 = (c31) / c11;
  c41 = (c41) / c11;
  c22 = c22 - c11 * (c21 * c21);
  c32 = (c32 - c11 * (c31 * c21)) / c22;
  c42 = (c42 - c11 * (c41 * c21)) / c22;
  c33 = c33 - c11 * (c31 * c31) - c22 * (c32 * c32);
  c43 = (c43 - c11 * (c41 * c31) - c22 * (c42 * c32)) / c33;
  c44 = c44 - c11 * (c41 * c41) - c22 * (c42 * c42) - c33 * (c43 * c43);
  real a11 = 1;
  real a12 = 0;
  real a13 = 0;
  real a14 = 0;
  real a21 = 0;
  real a22 = 1;
  real a23 = 0;
  real a24 = 0;
  real a31 = 0;
  real a32 = 0;
  real a33 = 1;
  real a34 = 0;
  real a41 = 0;
  real a42 = 0;
  real a43 = 0;
  real a44 = 1;
  a21 = a21 - a11 * c21;
  a22 = a22 - a12 * c21;
  a31 = a31 - a11 * c31;
  a32 = a32 - a12 * c31;
  a33 = a33 - a13 * c31;
  a31 = a31 - a21 * c32;
  a32 = a32 - a22 * c32;
  a33 = a33 - a23 * c32;
  a41 = a41 - a11 * c41;
  a42 = a42 - a12 * c41;
  a43 = a43 - a13 * c41;
  a44 = a44 - a14 * c41;
  a41 = a41 - a21 * c42;
  a42 = a42 - a22 * c42;
  a43 = a43 - a23 * c42;
  a44 = a44 - a24 * c42;
  a41 = a41 - a31 * c43;
  a42 = a42 - a32 * c43;
  a43 = a43 - a33 * c43;
  a44 = a44 - a34 * c43;
  a11 = a11 / c11;
  a21 = a21 / c22;
  a22 = a22 / c22;
  a31 = a31 / c33;
  a32 = a32 / c33;
  a33 = a33 / c33;
  a41 = a41 / c44;
  a42 = a42 / c44;
  a43 = a43 / c44;
  a44 = a44 / c44;
  a31 = a31 - a41 * c43;
  a32 = a32 - a42 * c43;
  a33 = a33 - a43 * c43;
  a21 = a21 - a31 * c32;
  a22 = a22 - a32 * c32;
  a21 = a21 - a41 * c42;
  a22 = a22 - a42 * c42;
  a11 = a11 - a21 * c21;
  a11 = a11 - a31 * c31;
  a11 = a11 - a41 * c41;
  a[0][0] = a11;
  a[0][1] = a21;
  a[0][2] = a31;
  a[0][3] = a41;
  a[1][0] = a21;
  a[1][1] = a22;
  a[1][2] = a32;
  a[1][3] = a42;
  a[2][0] = a31;
  a[2][1] = a32;
  a[2][2] = a33;
  a[2][3] = a43;
  a[3][0] = a41;
  a[3][1] = a42;
  a[3][2] = a43;
  a[3][3] = a44;
}

__device__ inline void invCholesky(real a[10]) {
  float c11 = a[0];
  float c21 = a[1];
  float c22 = a[2];
  float c31 = a[3];
  float c32 = a[4];
  float c33 = a[5];
  float c41 = a[6];
  float c42 = a[7];
  float c43 = a[8];
  float c44 = a[9];
  c21 = (c21) / c11;
  c31 = (c31) / c11;
  c41 = (c41) / c11;
  c22 = c22 - c11 * (c21 * c21);
  c32 = (c32 - c11 * (c31 * c21)) / c22;
  c42 = (c42 - c11 * (c41 * c21)) / c22;
  c33 = c33 - c11 * (c31 * c31) - c22 * (c32 * c32);
  c43 = (c43 - c11 * (c41 * c31) - c22 * (c42 * c32)) / c33;
  c44 = c44 - c11 * (c41 * c41) - c22 * (c42 * c42) - c33 * (c43 * c43);
  real a11 = 1;
  real a12 = 0;
  real a13 = 0;
  real a14 = 0;
  real a21 = 0;
  real a22 = 1;
  real a23 = 0;
  real a24 = 0;
  real a31 = 0;
  real a32 = 0;
  real a33 = 1;
  real a34 = 0;
  real a41 = 0;
  real a42 = 0;
  real a43 = 0;
  real a44 = 1;
  a21 = a21 - a11 * c21;
  a22 = a22 - a12 * c21;
  a31 = a31 - a11 * c31;
  a32 = a32 - a12 * c31;
  a33 = a33 - a13 * c31;
  a31 = a31 - a21 * c32;
  a32 = a32 - a22 * c32;
  a33 = a33 - a23 * c32;
  a41 = a41 - a11 * c41;
  a42 = a42 - a12 * c41;
  a43 = a43 - a13 * c41;
  a44 = a44 - a14 * c41;
  a41 = a41 - a21 * c42;
  a42 = a42 - a22 * c42;
  a43 = a43 - a23 * c42;
  a44 = a44 - a24 * c42;
  a41 = a41 - a31 * c43;
  a42 = a42 - a32 * c43;
  a43 = a43 - a33 * c43;
  a44 = a44 - a34 * c43;
  a11 = a11 / c11;
  a21 = a21 / c22;
  a22 = a22 / c22;
  a31 = a31 / c33;
  a32 = a32 / c33;
  a33 = a33 / c33;
  a41 = a41 / c44;
  a42 = a42 / c44;
  a43 = a43 / c44;
  a44 = a44 / c44;
  a31 = a31 - a41 * c43;
  a32 = a32 - a42 * c43;
  a33 = a33 - a43 * c43;
  a21 = a21 - a31 * c32;
  a22 = a22 - a32 * c32;
  a21 = a21 - a41 * c42;
  a22 = a22 - a42 * c42;
  a11 = a11 - a21 * c21;
  a11 = a11 - a31 * c31;
  a11 = a11 - a41 * c41;
  a[0] = a11;
  a[1] = a21;
  a[2] = a22;
  a[3] = a31;
  a[4] = a32;
  a[5] = a33;
  a[6] = a41;
  a[7] = a42;
  a[8] = a43;
  a[9] = a44;
}

// ----------------------------------------------------------------------
// eukildean norm
// s = ||x||_2

template <int N>
__device__ inline float norm_0 (float x[N]) {
  float s = 0.f;
  for (int k=0; k<N; k++)
    s += sqr (x[k]);
  return sqrtf (s);
}

__device__ inline float norm (float x[4]) {
  return sqrtf (sqr(x[0])+sqr(x[1])+sqr(x[2])+sqr(x[3]));
}

// ----------------------------------------------------------------------
// scalar product
// s = <x,y>

template <int N>
__device__ inline float scalProd_0 (float x[N], float y[N]) {
  float s = 0.f;
  for (int k=0; k<N; k++)
    s += x[k] * y[k];
  return s;
}

__device__ inline float scalProd (float x[4], float y[4]) {
  return x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
}

// ----------------------------------------------------------------------
// symmetric matrix A
// y = A*x

__device__ inline void matVec(const float a[10], const float x[4], float y[4]) {
  y[0] = a[0] * x[0] + a[1] * x[1] + a[3] * x[2] + a[6] * x[3];
  y[1] = a[1] * x[0] + a[2] * x[1] + a[4] * x[2] + a[7] * x[3];
  y[2] = a[3] * x[0] + a[4] * x[1] + a[5] * x[2] + a[8] * x[3];
  y[3] = a[6] * x[0] + a[7] * x[1] + a[8] * x[2] + a[9] * x[3];
}

__device__ inline void matVec(const float a[4][4], const float x[4], float y[4]) {
  for (int i=0; i<4; i++)
    y[i] = a[i][0] * x[0] + a[i][1] * x[1] + a[i][2] * x[2] + a[i][3] * x[3];
}

template <int N>
__device__ inline void matVec_0(const float a[N][N], const float x[N], float y[N]) {
  for (int i=0; i<N; i++) {
    float s = 0.f;
    for (int j=0; j<N; j++)
      s += a[i][j] * x[j];
    y[i] = s;
  }
}

// ----------------------------------------------------------------------
// symmetric matrix A, index 1..N-1, without index 0
// partial y = A*x

__device__ inline void matVec_1N(const float a[10], const float x[4], float y[4]) {
  y[0] = a[1] * x[1] + a[3] * x[2] + a[6] * x[3];
  y[1] = a[2] * x[1] + a[4] * x[2] + a[7] * x[3];
  y[2] = a[4] * x[1] + a[5] * x[2] + a[8] * x[3];
  y[3] = a[7] * x[1] + a[8] * x[2] + a[9] * x[3];
}

template <int N>
__device__ inline void matVec_1N(const float a[N][N], const float x[N], float y[N]) {
  for (int i=0; i<N; i++) {
    float s = 0.f;
    for (int j=1; j<N; j++)
      s += a[i][j] * x[j];
    y[i] = s;
  }
}

//----------------------------------------------------------------------

#endif // CHOLESKY_H
