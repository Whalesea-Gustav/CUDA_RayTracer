#pragma once

#include <cstdint>

class Window
{
public:

    static int Height, Width;
    static int8_t* Data;

    static void Init(int init_wdith, int init_height);

};