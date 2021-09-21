#pragma once

#include <Layers/Layer.h>

namespace mia
{
    namespace Tests
    {
        class LayerManipulator
        {
        public:
            static Matrix & GetWeightsMatrix(Layer & layer);
            static Matrix & GetBiasesMatrix(Layer & layer);
            static Matrix & GetValuesMatrix(Layer & layer);
        };
    }
}
