/*
** BENSUPERPC PROJECT, 2020
** RPG
** File description:
** screen_save.c
*/

#include "screen_save.h"

void saveScreenshotToFile_c(const char *filename, const int windowWidth, const int windowHeight)
{
    const int numberOfPixels = windowWidth * windowHeight * 3;
    GLubyte *pixels = malloc(numberOfPixels);
    if (pixels == NULL) {
        printf("malloc FAILED, can't save image\n");
        return;
    }

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, windowWidth, windowHeight, GL_BGR_EXT, GL_UNSIGNED_BYTE, pixels);

    FILE *outputFile = fopen(filename, "w");
    short header[] = {0, 2, 0, 0, 0, 0, (short)windowWidth, (short)windowHeight, 24};

    fwrite(&header, sizeof(header), 1, outputFile);
    fwrite(pixels, numberOfPixels, 1, outputFile);
    fclose(outputFile);
    free(pixels);
}