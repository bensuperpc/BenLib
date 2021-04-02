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
/**
 * @brief texture
 * @namespace texture
 */
namespace texture
{
/**
 * @brief 
 * 
 * @param xx Pos X
 * @param yy Pos y
 * @param a Alpha value
 * @param r Red value
 * @param g Green value
 * @param b Bleu value
 * @return sf::Texture 
 */
sf::Texture uniform_32(int &xx, int &yy, uint8_t a, uint8_t r, uint8_t g, uint8_t b);

/**
 * @brief 
 * 
 * @param xx Pos X
 * @param yy Pos y
 * @param r Red value
 * @param g Green value
 * @param b Bleu value
 * @return sf::Texture 
 */
sf::Texture uniform_24(int &xx, int &yy, uint8_t r, uint8_t g, uint8_t b);
} // namespace texture
} // namespace my
#endif
