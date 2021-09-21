#pragma once

#include "Models/Model.h"

namespace mia
{
    class Layer;

    // A Sequential Model represents an 1 dimensional array of layers where each
    // layer is connected to previous layer with exception of the first layer. The
    // first layer is expected to be an InputLayer.
    class Sequential : public Model
    {
    public:
        Sequential() = delete;
        virtual ~Sequential();

        // Creates a Sequential Model using the supplied list of heap-allocated layers.
        Sequential(std::initializer_list<Layer *> const & layers);

        virtual void Compile(u32 seedValue) override;
        virtual void Train(NDArrayView<f32> const & inputData, std::initializer_list<f32> const & expectedOutput) override;

    private:
        void ForwardPropagation();

    private:
        static u32 constexpr c_MaxNumLayers = 256;

        u32 m_NumLayers;
        Layer * m_Layers[c_MaxNumLayers];
    };
}
