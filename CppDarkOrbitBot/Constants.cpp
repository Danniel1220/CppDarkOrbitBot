#include "Constants.h"

#include <windows.h>

HANDLE consoleHandle = nullptr;

void initializeConsoleHandle() 
{
    consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
}