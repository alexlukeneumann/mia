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
            ReLU,
            Sigmoid
        };

        typedef f32 (*Activator)(f32 x);
        static Activator GetActivator(ActivatorType type)
        {
            switch (type)
            {
                case ActivatorType::ReLU:       return ReLU;
                case ActivatorType::Sigmoid:    return Sigmoid;

                default:
                    ASSERTMSG(false, "Unknown ActivatorType.");
                    break;
            }

            return nullptr;
        }

        typedef f32 (*ActivatorDerivative)(f32 x);
        static ActivatorDerivative GetActivatorDerivative(ActivatorType type)
        {
            switch (type)
            {
                case ActivatorType::ReLU:       return ReLUDerivative;
                case ActivatorType::Sigmoid:    return SigmoidDerivative;

                default:
                    ASSERTMSG(false, "Unknown ActivatorType.");
                    break;
            }

            return nullptr;
        }
    }
}
