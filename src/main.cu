#include <iostream>
#include "DeviceManager.h"
#include "util/logger.h"

int main()
{
    using namespace gprender;

    GPLog::init();

    PrintDeviceInfo();
}