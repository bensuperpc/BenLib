/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** screen_save.cpp
*/

#include "screen_save.hpp"

////////////////////////////////////////////////////////////
/// \brief set the position of the object
///
/// This function can save screenshot from openGL to file.
/// It use OpenCV or fwrite
/// OK :)
///
/// \param filename File name, if use .tga, you use fwrite, other OpenCV
/// \param windowWidth Window size X
/// \param windowHeight Window size Y
///
/// \see ?
///
////////////////////////////////////////////////////////////
void Screen_save_gl::saveScreenshotToFile(const std::string &filename, const size_t &windowWidth, const size_t &windowHeight)
{
    // Thank https://stackoverflow.com/a/53110929/10152334
    // https://stackoverflow.com/questions/9097756/converting-data-from-glreadpixels-to-opencvmat/9098883
    // https://stackoverflow.com/questions/2340281/check-if-a-string-contains-a-string-in-c
    if (filename.find(".tga") != std::string::npos) {
        saveScreenshotToFile_c(filename.c_str(), windowWidth, windowHeight);
    } else {
        // cv::Mat &&img = saveScreenshotToMat(filename, windowWidth, windowHeight);
        cv::Mat img = cv::Mat(windowHeight, windowWidth, CV_8UC4);
        saveScreenshotToMat(filename, windowWidth, windowHeight, img);
        cv::imwrite(filename, img);
        img.release();
#ifndef DNDEBUG
        std::cout << "Save IMG: OK" << std::endl;
#endif
    }
}

inline cv::Mat Screen_save_gl::saveScreenshotToMat(const std::string &filename, const size_t &windowWidth, const size_t &windowHeight)
{
    cv::Mat img(windowHeight, windowWidth, CV_8UC4);
    glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
    glPixelStorei(GL_PACK_ROW_LENGTH, img.step / img.elemSize());
    glReadPixels(0, 0, img.cols, img.rows, GL_BGRA, GL_UNSIGNED_BYTE, img.data);
    cv::flip(img, img, 0);
    return img;
}

void Screen_save_gl::saveScreenshotToMat(const std::string &filename, const size_t &windowWidth, const size_t &windowHeight, cv::Mat &img)
{
    glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
    glPixelStorei(GL_PACK_ROW_LENGTH, img.step / img.elemSize());
    glReadPixels(0, 0, img.cols, img.rows, GL_BGRA, GL_UNSIGNED_BYTE, img.data);
    cv::flip(img, img, 0);
}