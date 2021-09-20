#include "LayerManipulator.h"

namespace mia
{
    namespace Tests
    {
        Matrix & LayerManipulator::GetWeightsMatrix(Layer & layer)
        {
            return layer.m_Weights;
        }

        Matrix & LayerManipulator::GetValuesMatrix(Layer & layer)
        {
            return layer.m_Values;
        }
    }
}
