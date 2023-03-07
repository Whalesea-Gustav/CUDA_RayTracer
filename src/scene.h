#pragma once

#include "camera.h"

class Scene
{
public:
    static Scene* instance;

    Camera* m_camera;

    static Scene* Instance();

};