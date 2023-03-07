#include "window.h"

int Window::Height = 0;
int Window::Width = 0;
int8_t* Window::Data = nullptr;

void Window::Init(int init_wdith, int init_height) {
    Width = init_wdith;
    Height = init_height;
    Data = new int8_t[Width * Height * 4];
}
