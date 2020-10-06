/*
** Bensuperpc, 2016-2019
** Servolent 2016 and Jeu de réflexe 2017
** BAC project 2016-2017
** File description:
** >reboot_software.c
*/

void reboot_software(void)
{
    wdt_enable(WDTO_15MS);
}
