#pragma once

#include "Layers/InputLayer.h"

namespace mia
{
    class Flatten final : public InputLayer
    {
    public:
        Flatten();
        ~Flatten();

        // Constructs a Flatten layer using the supplied input shape.
        // e.g. { 2, 2, 2 } describes a 2x2x2 dataset.
        Flatten(std::initializer_list<DimensionLength> const & inputDimensionLengths);

        virtual void Compile(Layer const * prevLayer) override;
        virtual void Execute(Layer const * prevLayer) override { /* Flattening does not have weights associated with it. */ }

        // Converts & flattens the supplied inputData into the local m_Values matrix for use by another layer.
        virtual void SetInputData(NDArrayView<f32> const & inputData) override;

    private:
        u32 m_InputNumDimensions;
        DimensionLength * m_InputDimensionLengths;
    };
}
