#pragma once

#include "Common.h"

#include "Activators/ReLU.h"
#include "Activators/Sigmoid.h"

namespace mia
{
    namespace activators
    {
        enum class ActivatorType : u8
        {
            None,
            ReLU,
            Sigmoid
        };

        typedef f32 (*Activator)(f32 x);
        static Activator GetActivator(ActivatorType type)
        {
            switch (type)
            {
                case ActivatorType::None:       return nullptr;
                case ActivatorType::ReLU:       return ReLU;
                case ActivatorType::Sigmoid:    return Sigmoid;

                default:
                    ASSERTMSG(false, "Unknown ActivatorType.");
                    break;
            }

            return nullptr;
        }
    }
}
