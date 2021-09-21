#pragma once

#include "Layers/Layer.h"

namespace mia
{
    // Dense layer is responsible for connecting every neuron in the previous layer
    // to each neuron in this layer.
    class Dense final : public Layer
    {
    public:
        Dense() = delete;
        virtual ~Dense() = default;

        // Constructs a Dense layer with n number of neurons contained within
        Dense(u32 numNeurons);
        
        virtual void Compile(u32 seedValue, Layer const * prevLayer) override;

    private:
        u32 m_NumNeurons;
    };
}
