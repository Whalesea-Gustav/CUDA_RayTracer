#include <iostream>
#include "DeviceManager.h"
#include "util/logger.h"
#include "window.h"
#include "scene.h"

int main()
{
    using namespace gprender;

    GPLog::init();

    PrintDeviceInfo();

    Scene::Instance();

}