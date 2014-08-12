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

#define N 4
#define real float
__device__ inline void factorSubstCholesky0(real a[N][N], real b[N]) {
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


//----------------------------------------------------------------------

#endif // CHOLESKY_H
