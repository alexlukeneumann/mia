#pragma once

#include "Layers/InputLayer.h"

namespace mia
{
    class Flatten final : public InputLayer
    {
    public:
        Flatten();
        ~Flatten();

        // Constructs a Flatten layer than expects inputNumDimensions number of dimensions with
        // an array of DimensionLengths that must be the same length as inputNumDimensions
        Flatten(u32 inputNumDimensions, DimensionLength * inputDimensionLengths);

        virtual void Compile(Layer const * prevLayer) override;
        virtual void Execute(Layer const * prevLayer) override { /* Flattening does not have weights associated with it. */ }

        // Converts & flattens the supplied inputData into the local m_Values matrix for use by another layer.
        virtual void SetInputData(NDArrayView<f32> const & inputData) override;

    private:
        u32 m_InputNumDimensions;
        DimensionLength * m_InputDimensionLengths;
    };
}
