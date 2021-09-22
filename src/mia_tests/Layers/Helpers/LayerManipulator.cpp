#include "LayerManipulator.h"

namespace mia
{
    namespace Tests
    {
        Matrix & LayerManipulator::GetWeightsMatrix(layers::Layer & layer)
        {
            return layer.m_Weights;
        }

        Matrix & LayerManipulator::GetBiasesMatrix(layers::Layer & layer)
        {
            return layer.m_Biases;
        }

        Matrix & LayerManipulator::GetValuesMatrix(layers::Layer & layer)
        {
            return layer.m_Values;
        }
    }
}
