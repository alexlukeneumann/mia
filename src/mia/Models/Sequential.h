#pragma once

#include "Models/Model.h"

namespace mia
{
    namespace layers
    {
        class Layer;
    }

    namespace models
    {
        // A Sequential Model represents an 1 dimensional array of layers where each
        // layer is connected to previous layer with exception of the first layer. The
        // first layer is expected to be an InputLayer.
        class Sequential : public Model
        {
        public:
            Sequential() = delete;
            virtual ~Sequential();

            // Creates a Sequential Model using the supplied list of heap-allocated layers.
            Sequential(std::initializer_list<layers::Layer *> const & layers);

            virtual void Compile(u32 seedValue) override;
            virtual f32 Train(NDArrayView<f32> const & inputData, std::initializer_list<f32> const & expectedOutput) override;
            virtual Matrix Execute(NDArrayView<f32> const & inputData) override;

        private:
            void ForwardPropagation();
            void BackPropagation(std::initializer_list<f32> const & expectedOutput);

        private:
            static u32 constexpr c_MaxNumLayers = 256;

            u32 m_NumLayers;
            layers::Layer * m_Layers[c_MaxNumLayers];
        };
    }
}
