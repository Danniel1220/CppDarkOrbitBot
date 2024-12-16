#ifndef CONSTANTS
#define CONSTANTS

#include <windows.h>

extern HANDLE consoleHandle;

// 0 = Black     8 = Gray
// 1 = Blue      9 = Light Blue
// 2 = Green     a = Light Green
// 3 = Aqua      b = Light Aqua
// 4 = Red       c = Light Red
// 5 = Purple    d = Light Purple
// 6 = Yellow    e = Light Yellow
// 7 = White     f = Bright White

constexpr int BLUE_TEXT_BLACK_BACKGROUND = 1;
constexpr int GREEN_TEXT_BLACK_BACKGROUND = 2;
constexpr int RED_TEXT_BLACK_BACKGROUND = 4;
constexpr int YELLOW_TEXT_BLACK_BACKGROUND = 6;
constexpr int DEFAULT = 15;

void initializeConsoleHandle();

#endif
