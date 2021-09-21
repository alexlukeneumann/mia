#pragma once

#include <Layers/InputLayer.h>

namespace mia
{
    namespace Tests
    {
        class TestInputLayer final : public InputLayer
        {
        public:
            TestInputLayer() = delete;
            TestInputLayer(u32 numNeurons);
            virtual ~TestInputLayer() = default;

            virtual void Compile(u32 seedValue, Layer const * prevLayer) override;
            virtual void SetInputData(NDArrayView<f32> const & inputData) override;

        private:
            u32 m_NumNeurons;
        };
    }
}
