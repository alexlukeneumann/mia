#pragma once

#include "Core/NDArrayView.h"
#include "Layers/Layer.h"

namespace mia
{
    namespace layers
    {
        class InputLayer : public Layer
        {
        public:
            InputLayer(activators::ActivatorType activatorType)
                : Layer(activatorType)
            {}

            virtual ~InputLayer() = default;

            virtual LayerType GetType() const override { return LayerType::Input; }

            // Converts the supplied inputData into the local m_Values matrix for use by another layer.
            virtual void SetInputData(NDArrayView<f32> const & inputData) = 0;
        };
    }
}
