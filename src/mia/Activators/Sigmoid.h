#pragma once

#include "Common.h"

namespace mia
{
    namespace activators
    {
        // Sigmoid
        // 
        // Represents a characteristic "S" shaped curve about zero.
        // The returned value of this function is between 0 & 1. As this
        // activator is non-linear, it is great to use in networks
        // where the problem is complex like XOR logic or patterns seperated
        // into curves/circles.
        // 
        // For a large network, this activator can suffer from the "gradient vanishing"
        // problem.
        static f32 Sigmoid(f32 x)
        {
            return 1 / (1 + exp(-x));
        }
    }
}
