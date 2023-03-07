#include "scene.h"

Scene* Scene::instance = nullptr;

Scene* Scene::Instance()
{
    if (instance == nullptr)
    {
        instance = new Scene();
    }
    return instance;
}
