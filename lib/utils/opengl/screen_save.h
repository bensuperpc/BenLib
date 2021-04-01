/**
 * @file screen_save.h
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief 
 * @version 1.0.0
 * @date 2021-04-01
 * 
 * MIT License
 * 
 */

#ifndef _SCREEN_SAVE_GL_H_
#define _SCREEN_SAVE_GL_H_

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief 
 * 
 * @param filename 
 * @param windowWidth 
 * @param windowHeight 
 */
void saveScreenshotToFile_c(const char *filename, const int windowWidth, const int windowHeight);
#endif
