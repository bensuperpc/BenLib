//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  BenLib, 2020                                            //
//  Created: 11, Novemver, 2020                             //
//  Modified: 11, Novemver, 2020                            //
//  file: texyure.cpp                                       //
//  Load link titles                                        //
//  Source: -                                               //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include "texture.hpp"

sf::Texture my::texture::uniform_32(int &xx, int &yy, uint8_t a, uint8_t r, uint8_t g, uint8_t b)
{
    sf::Texture texture;
    texture.create((unsigned int)xx, (unsigned int)yy);
    sf::Uint8  *pixels  = new sf::Uint8[xx * yy * 4];

    for(int x = 0; x < xx; x++)
    {
        for(int y = 0; y < yy; y++)
        {
            pixels[(x + y * xx) * 4]     = r; // Red
            pixels[(x + y * xx) * 4 + 1] = g; // Green
            pixels[(x + y * xx) * 4 + 2] = b; // Blue
            pixels[(x + y * xx) * 4 + 3] = a; // Alpha
        }
    }
    texture.update(pixels);
    delete [] pixels;
    return texture;
}

sf::Texture my::texture::uniform_24(int &xx, int &yy, uint8_t r, uint8_t g, uint8_t b)
{
    sf::Texture texture;
    texture.create((unsigned int)xx, (unsigned int)yy);
    sf::Uint8  *pixels  = new sf::Uint8[xx * yy * 3];

    for(int x = 0; x < xx; x++)
    {
        for(int y = 0; y < yy; y++)
        {
            pixels[(x + y * xx) * 4]     = r; // Red
            pixels[(x + y * xx) * 4 + 1] = g; // Green
            pixels[(x + y * xx) * 4 + 2] = b; // Blue
        }
    }
    texture.update(pixels);
    delete [] pixels;
    return texture;
}