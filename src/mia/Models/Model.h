#pragma once

#include "Common.h"
#include "Core/NDArrayView.h"

namespace mia
{
    namespace models
    {
        class Model
        {
        public:
            Model() = default;
            Model(Model const & other) = delete;
            Model(Model && other) = delete;
            virtual ~Model() = default;

            // Compiles the current model so that it is ready to be trained or
            // executed.
            virtual void Compile(u32 seedValue) = 0;

            // Executes the current state of the model on the supplied inputData and then
            // adjusts the trainable parameters based on how close the output result is to
            // the supplied expectedOutput.
            virtual void Train(NDArrayView<f32> const & inputData, std::initializer_list<f32> const & expectedOutput) = 0;
        };
    }
}
