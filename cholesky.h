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

//----------------------------------------------------------------------
#define N 4
#define real float
#define DEVICE __device__


//----------------------
// faster, 16 values per matrix

DEVICE void factorCholesky_sym2(real a[N][N], real id[N]) {
  // LDLt Cholesky decomposition
  // L in a
  // D in a
  // inv D in id
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
      for (int k=0; k<j; k++)
	s -= a[k][k] * a[i][k] * a[j][k]; 
      a[i][j] = s * id[j];
    }
  }
}

DEVICE void invCholesky_sym2(real a[N][N]) {
  real c[N][N], id[N];

  for (int i=0; i<N; i++)
    for (int j=0; j<=i; j++)
      c[i][j] = a[i][j];

  factorCholesky_sym2(c, id);

  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
      a[i][j] = 0.f;
  for (int i=0; i<N; i++)
      a[i][i] = 1.f;

  for (int i=1; i<N; i++) {
    for (int k=0; k<i; k++)
      for (int j=0; j<=i; j++)
	a[i][j] -= a[k][j] * c[i][k];
  }
  for (int i=0; i<N; i++)
    for (int j=0; j<=i; j++)
      a[i][j] *= id[i];
  for (int i=N-2; i>=0; i--) {
    for (int k=i+1; k<N; k++)
      for (int j=0; j<=i; j++)
	a[i][j] -= a[k][j] * c[k][i];
  }

  for (int i=1; i<N; i++)
    for (int j=0; j<i; j++)
      a[j][i] = a[i][j];
}

#undef N
//----------------------------------------------------------------------

#endif // CHOLESKY_H
