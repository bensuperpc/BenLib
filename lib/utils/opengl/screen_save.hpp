/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** screen_save.hpp
*/

#ifndef _SCREEN_SAVE_GL_HPP_
#define _SCREEN_SAVE_GL_HPP_

// Disable Warning from OpenCV libs
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wfloat-equal"
#pragma GCC diagnostic ignored "-Wpadded"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#pragma GCC diagnostic ignored "-Wswitch-default"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#pragma GCC diagnostic pop

#include <string>

extern "C"
{
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include "screen_save.h"
}

namespace Screen_save_gl
{
void saveScreenshotToFile(const std::string &, const size_t &, const size_t &);
cv::Mat saveScreenshotToMat(const std::string &, const size_t &, const size_t &);
void saveScreenshotToMat(const std::string &, const size_t &, const size_t &, cv::Mat &);
} // namespace Screen_save_gl

#endif
