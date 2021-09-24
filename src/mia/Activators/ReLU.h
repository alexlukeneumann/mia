#pragma once

#include "Common.h"

namespace mia
{
    namespace activators
    {
        // ReLU
        // 
        // The rectified linear activation function or ReLU for short is a piecewise
        // linear function that will output the input if it is positive, otherwise,
        // it will output zero.
        //
        // ReLU overcomes the "vanishing gradient" problem, allowing models to learn
        // faster and perform better.
        static f32 ReLU(f32 x)
        {
            return std::max(0.0f, x);
        }

        static f32 ReLUDerivative(f32 x)
        {
            return (x > 0.0f) ? 1.0f : 0.0f;
        }
    }
}
