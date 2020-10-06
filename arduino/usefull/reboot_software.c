/*
** Bensuperpc, 2016-2019
** -
** File description:
** >reboot_software.c
*/

void reboot_software(void) {
  wdt_enable(WDTO_15MS);
}
