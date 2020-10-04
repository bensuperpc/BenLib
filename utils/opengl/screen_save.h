/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** screen_save.hpp
*/

#ifndef _SCREEN_SAVE_GL_H_
#define _SCREEN_SAVE_GL_H_

#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>

void saveScreenshotToFile_c(const char *filename, const int windowWidth, const int windowHeight);
#endif
