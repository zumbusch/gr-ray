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

// modified version of cpu_anim.h
// CUDA by Example: An Introduction to General-Purpose GPU Programming
// July 2010


#ifndef __CPU_ANIM_H__
#define __CPU_ANIM_H__

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
      printf( "%s(%i) in %s at line %d\n", cudaGetErrorString(err), (int)err,
	      file, line );
      exit (EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
	      file, line, errorMessage, (int)err, cudaGetErrorString(err));
      cudaDeviceReset ();
      exit (EXIT_FAILURE);
    }
}


#include "gl_helper.h"

#include <iostream>


struct CPUAnimBitmap {
  unsigned char *pixels;
  int width, height;
  void *dataBlock;
  void (*reshapeC)(void*,int,int);
  void (*fAnim)(void*);
  void (*animKey)(void*,unsigned char);
  void (*animExit)(void*);
  void (*clickDrag)(void*,float,float,float,float,float);
  int dragStartX, dragStartY;
  int mouseOldX, mouseOldY, mouseButtons, modifiers;
  float rotateX, rotateY, translateX, translateY, translateZ;
  int update;

  CPUAnimBitmap( int w, int h, void *d = NULL ) {
    width = w;
    height = h;
    pixels = new unsigned char[width * height * 4];
    dataBlock = d;
    clickDrag = NULL;
    mouseOldX = mouseOldY = mouseButtons = 0;
    rotateX = rotateY = translateX = translateY = translateZ = 0;
    update = 0;
  }

  ~CPUAnimBitmap() {
    delete [] pixels;
  }

  unsigned char* get_ptr( void ) const   { return pixels; }
  long image_size( void ) const { return width * height * 4; }

  void click_drag( void (*f)(void*,float,float,float,float,float),
		   void (*c)(void*,int,int)) {
    clickDrag = f;
    reshapeC = c;
  }

  void anim_and_exit( void (*f)(void*), void(*e)(void*), void(*t)(void*,unsigned char)) {
    CPUAnimBitmap** bitmap = get_bitmap_ptr();
    *bitmap = this;
    fAnim = f;
    animExit = e;
    animKey = t;
    // a bug in the Windows GLUT implementation prevents us from
    // passing zero arguments to glutInit()
    int c=1;
    //char* dummy = "";
    //glutInit( &c, &dummy );
    glutInit( &c, 0 );
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
    glutInitWindowSize( width, height );
    glutCreateWindow( "GR raytracer" );
    glutKeyboardFunc(Key);
    glutDisplayFunc(Draw);
    glutMouseFunc( mouse_func );
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);
    glutIdleFunc( idle_func );
    glutMainLoop();
  }

  // static method used for glut callbacks
  static CPUAnimBitmap** get_bitmap_ptr( void ) {
    static CPUAnimBitmap*   gBitmap;
    return &gBitmap;
  }

  // static method used for glut callbacks
  static void mouse_func(int button, int state, int x, int y) {
    CPUAnimBitmap* bitmap = *(get_bitmap_ptr());
    if (state == GLUT_DOWN) {
      bitmap->mouseButtons |= 1<<button;
    } else if (state == GLUT_UP) {
      bitmap->mouseButtons = 0;
    }
    bitmap->modifiers = glutGetModifiers();

    bitmap->mouseOldX = x;
    bitmap->mouseOldY = y;
    glutPostRedisplay();
  }

  static void motion(int x, int y) {
    CPUAnimBitmap* bitmap = *(get_bitmap_ptr());
    float dx, dy;
    dx = (float)(x - bitmap->mouseOldX);
    dy = (float)(y - bitmap->mouseOldY);
    int mods = bitmap->modifiers;
    if (((mods & GLUT_ACTIVE_SHIFT) && bitmap->mouseButtons == 1)
	|| (bitmap->mouseButtons == 2)) {
      bitmap->rotateX += dx * 0.01f;
      bitmap->rotateY -= dy * 0.01f;
      //printf("rot ");
    } else {
      if (bitmap->mouseButtons == 1) {
	bitmap->translateX += dx * 0.01f;
	bitmap->translateY -= dy * 0.01f;        
	//printf("trans ");
      } else if (bitmap->mouseButtons == 4) {
	bitmap->translateZ += dy * 0.01f;
	//printf("z ");
      }
    }
    bitmap->mouseOldX = x;
    bitmap->mouseOldY = y;
    bitmap->clickDrag( bitmap->dataBlock, bitmap->rotateX, bitmap->rotateY, bitmap->translateX, bitmap->translateY, bitmap->translateZ);
    bitmap->update = 1;
  }

  static void reshape(int w, int h) {
    CPUAnimBitmap* bitmap = *(get_bitmap_ptr());
    int wx = 32, hx = 16;
    w = ((w + wx -1)/wx)*wx;
    h = ((h + hx -1)/hx)*hx;
    bitmap->width = w;
    bitmap->height = h;
    if (bitmap->pixels)
      delete[] bitmap->pixels;
    bitmap->pixels = new unsigned char[w * h * 4];
    bitmap->reshapeC(  bitmap->dataBlock, w, h );
    bitmap->update = 1;
  }

  // static method used for glut callbacks
  static void idle_func( void ) {
    CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
    bitmap->fAnim( bitmap->dataBlock );
    glutPostRedisplay();
  }

  // static method used for glut callbacks
  static void Key(unsigned char key, int x, int y) {
    CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
    switch (key) {
    case char('M'):
    case char('m'):
    case char('e'):
      bitmap->animKey( bitmap->dataBlock, key );
      break;
    case 27:
      bitmap->animExit( bitmap->dataBlock );
      exit(0);
    }
    bitmap->update = 1;
  }

  // static method used for glut callbacks
  static void Draw( void ) {
    CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
    glClearColor( 0.0, 0.0, 0.0, 1.0 );
    glClear( GL_COLOR_BUFFER_BIT );
    glDrawPixels( bitmap->width, bitmap->height, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels );
    glutSwapBuffers();
  }
};


#endif  // __CPU_ANIM_H__
