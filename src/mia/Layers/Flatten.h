#pragma once

#include "Layers/InputLayer.h"

namespace mia
{
    namespace layers
    {
        // Flatten layer acts as an input layer to the neural network and converts a 
        // n-dimensional array of data into a single dimension.
        class Flatten final : public InputLayer
        {
        public:
            Flatten() = delete;
            virtual ~Flatten();

            // Constructs a Flatten layer using the supplied input shape.
            // e.g. { 2, 2, 2 } describes a 2x2x2 dataset.
            Flatten(std::initializer_list<DimensionLength> const & inputDimensionLengths, activators::ActivatorType activatorType = activators::ActivatorType::ReLU);

            virtual void Compile(u32 seedValue, Layer const * prevLayer) override;
            virtual void Execute(Layer const * prevLayer) override { /* Flattening does not have weights associated with it. */ }

            // Converts & flattens the supplied inputData into the local m_Values matrix for use by another layer.
            virtual void SetInputData(NDArrayView<f32> const & inputData) override;

        private:
            u32 m_InputNumDimensions;
            DimensionLength * m_InputDimensionLengths;
        };
    }
}
