#include "Constants.h"

#include <windows.h>
#include <iostream>

HANDLE consoleHandle = nullptr;

void initializeConsoleHandle() 
{
    consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
}