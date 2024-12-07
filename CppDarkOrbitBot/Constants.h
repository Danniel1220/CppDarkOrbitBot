#ifndef CONSTANTS
#define CONSTANTS

#include <windows.h>

extern HANDLE consoleHandle;

constexpr int GREEN_TEXT_BLACK_BACKGROUND = 2;
constexpr int RED_TEXT_BLACK_BACKGROUND = 4;
constexpr int YELLOW_TEXT_BLACK_BACKGROUND = 6;
constexpr int DEFAULT = 15;

void initializeConsoleHandle();

#endif
