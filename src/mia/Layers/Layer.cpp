#include "Layer.h"

namespace mia
{
    namespace layers
    {
        Layer::Layer(activators::ActivatorType activatorType)
            : m_ActivatorType(activatorType)
        {
        }

        void Layer::Execute(Layer const * prevLayer)
        {
            ASSERTMSG(nullptr != prevLayer, "prevLayer is not a valid ptr.");
            ASSERTMSG(GetNumNeurons() > 0, "layer doesn't have any neurons.");

            // Base implemention:
            // - Multiply the pre-filled m_Weights structure by the prevLayer's
            // already calculated m_Values structure.
            // - Add the m_Biases matrix to the m_Values structure
            // - Apply the activation function onto each computed neuron computed value
            //
            // The first two steps essentially summates the dot product of each neuron,
            // in the previous layer, connected to a particular neuron with its weight and
            // then adds the neuron's particular bias value to the final value for the neuron.
            // The final step helps determine whether the neuron should "activate" or not.

            m_ValuesPriorActivator = std::move(Matrix::Multiply(m_Weights, prevLayer->GetValues()));
            m_ValuesPriorActivator = std::move(Matrix::Add(m_ValuesPriorActivator, m_Biases));

            activators::Activator activator = activators::GetActivator(m_ActivatorType);
            if (nullptr != activator)
            {
                for (u32 rIdx = 0; rIdx < m_Values.GetHeight(); ++rIdx)
                {
                    f32 const & neuronValue = m_ValuesPriorActivator.GetElement(rIdx, 0);
                    f32 & finalNeuronValue = m_Values.GetElement(rIdx, 0);

                    finalNeuronValue = activator(neuronValue);
                }
            }
            else
            {
                m_Values = m_ValuesPriorActivator;
            }
        }
    }
}
