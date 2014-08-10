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

// ----------------------------------------------------------------------
// 
// ----------------------------------------------------------------------


__device__ void invCholesky(float a[4][4]) {
  float s;
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
  s = c11;
  c11 = s;
  s = c21;
  c21 = s / c11;
  s = c31;
  c31 = s / c11;
  s = c41;
  c41 = s / c11;
  s = c22;
  s = s - c11 * c21 * c21;
  c22 = s;
  s = c32;
  s = s - c11 * c31 * c21;
  c32 = s / c22;
  s = c42;
  s = s - c11 * c41 * c21;
  c42 = s / c22;
  s = c33;
  s = s - c11 * c31 * c31;
  s = s - c22 * c32 * c32;
  c33 = s;
  s = c43;
  s = s - c11 * c41 * c31;
  s = s - c22 * c42 * c32;
  c43 = s / c33;
  s = c44;
  s = s - c11 * c41 * c41;
  s = s - c22 * c42 * c42;
  s = s - c33 * c43 * c43;
  c44 = s;
  float a11 = 1.f;
  float a12 = 0.f;
  float a13 = 0.f;
  float a14 = 0.f;
  float a21 = 0.f;
  float a22 = 1.f;
  float a23 = 0.f;
  float a24 = 0.f;
  float a31 = 0.f;
  float a32 = 0.f;
  float a33 = 1.f;
  float a34 = 0.f;
  float a41 = 0.f;
  float a42 = 0.f;
  float a43 = 0.f;
  float a44 = 1.f;
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

//----------------------------------------------------------------------

#endif // CHOLESKY_H
