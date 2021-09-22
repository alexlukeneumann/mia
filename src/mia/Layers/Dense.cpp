#include "Dense.h"

namespace mia
{
    Dense::Dense(u32 numNeurons, activators::ActivatorType activatorType)
        : Layer(activatorType)
        , m_NumNeurons(numNeurons)
    {
    }

    void Dense::Compile(u32 seedValue, Layer const * prevLayer)
    {
        ASSERTMSG(nullptr != prevLayer, "Dense layer cannot be used without a preceeding layer.");
        ASSERTMSG(0 < prevLayer->GetNumNeurons(), "Previous layer doesn't have any neurons.");

        // Reserve space in the m_Values matrix for our neurons
        m_Values = Matrix(1, m_NumNeurons);

        // Reserve space in the m_Weights matrix & seed it
        m_Weights = Matrix(prevLayer->GetNumNeurons(), m_NumNeurons);
        m_Weights.Seed(seedValue);

        // Reserve space in the m_Biases matrix & seed it
        m_Biases = Matrix(1, m_NumNeurons);
        m_Biases.Seed(seedValue);
    }
}
