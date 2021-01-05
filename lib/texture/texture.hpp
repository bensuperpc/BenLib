//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  RPG, 2020                                               //
//  Created: 11, Novemver, 2020                             //
//  Modified: 11, Novemver, 2020                            //
//  file: load_titlemap.cpp                                 //
//  Load link titles                                        //
//  Source: -                                               //
//  OS: ALL                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef TEXTURE_HPP_
#define TEXTURE_HPP_

#include <SFML/Graphics.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
namespace my
{
namespace texture
{
sf::Texture uniform_32(int &, int &, uint8_t, uint8_t, uint8_t, uint8_t);
sf::Texture uniform_24(int &, int &, uint8_t, uint8_t, uint8_t);
} // namespace texture
} // namespace my
#endif
