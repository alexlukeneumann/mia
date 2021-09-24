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

        Matrix Layer::Backpropagation(Matrix const & expectedOutput, bool isOutputLayer, Layer const * prevLayer)
        {
            f32 const momentumRate = 0.03f;
            f32 const learningRate = 0.01f;

            // Retrieve the activation values of the output neurons
            Matrix const & outputNeuronValues = this->GetValues();

            // Calculate the "cost" of each output neuron.
            // Cost is the value associated with how much error there is in neuron's output
            // and the expected value. (dC0/dAj)
            //
            // Cost = (Activation Value - Expected Value) ^ 2
            //
            // For calculating how sensitive the cost function is w.r.t. weights/biases/activation value
            // for a neuron in a previous layer, we need the derivative of the function w.r.t. activation value.
            //
            // => 2 * (Activation Value - Expected Value)
            Matrix costSensitivities = Matrix(1, this->GetNumNeurons());
            if (isOutputLayer)
            {
                for (u32 rIdx = 0; rIdx < expectedOutput.GetHeight(); ++rIdx)
                {
                    f32 const & expected = expectedOutput.GetElement(rIdx, 0);
                    f32 const & outputted = outputNeuronValues.GetElement(rIdx, 0);

                    costSensitivities.GetElement(rIdx, 0) = 2 * (outputted - expected);
                }
            }
            else
            {
                // Already been calculated
                costSensitivities = expectedOutput;
            }

            // Calculate how sensitive the activation of each output neuron is to 
            // the dot product + bias function. (dAj/dZj)
            //
            // This is just the derivative of the activation function.
            Matrix activationSensitivities = Matrix(1, this->GetNumNeurons());
            {
                activators::ActivatorDerivative derivative = activators::GetActivatorDerivative(this->GetActivatorType());
                Matrix const & neuronValues = this->GetValuesPriorActivator();
                for (u32 rIdx = 0; rIdx < activationSensitivities.GetHeight(); ++rIdx)
                {
                    activationSensitivities.GetElement(rIdx, 0) = derivative(neuronValues.GetElement(rIdx, 0));
                }
            }

            // We can now calculate the sensitive the total cost, for a particular output neuron,
            // is to a particular weight & bias

            // Weights
            //
            // How sensitive the total cost, for a particular output neuron, w.r.t. a particular weight is related to
            // how sensitive the cost is to the activation value, how sensitive the activation
            // is to the dot product + bias function AND how sensitive the dot product + bias function 
            // is to the weight connecting a neuron to the particular output neuron.
            //
            // We have already calculated the first two, so to calculate the third term we just need
            // the activation value of the connected node in the previous layer (dZj/dWjk)
            Matrix const & outputNeuronWeights = this->GetWeights();
            Matrix weightGradients = Matrix(outputNeuronWeights.GetWidth(), outputNeuronWeights.GetHeight());
            {
                for (u32 rIdx = 0; rIdx < weightGradients.GetHeight(); ++rIdx)
                {
                    for (u32 cIdx = 0; cIdx < weightGradients.GetWidth(); ++cIdx)
                    {
                        // Retrieve the activation value of the neuron in the previous layer
                        // associated with this weight.
                        f32 const & neuronActivationValue = prevLayer->GetValues().GetElement(cIdx, 0);

                        f32 momentum = 0.0f;
                        if (m_PrevWeightGradients.GetCapacity() > 0)
                        {
                            momentum = m_PrevWeightGradients.GetElement(rIdx, cIdx) * momentumRate;
                        }

                        // Calculate the weight gradient (how sensitive the cost is to this particular weight)
                        weightGradients.GetElement(rIdx, cIdx) =
                            neuronActivationValue * activationSensitivities.GetElement(rIdx, 0) * costSensitivities.GetElement(rIdx, 0) + momentum;
                    }
                }
            }

            // Biases
            //
            // How sensitive the total cost, for a particular output neuron, w.r.t. a particular bias is related to 
            // how sensitive the cost is to the activation value, how sensitive the activation
            // is to the dot product + bias function AND how sensitive the dot product + bias function is to
            // the bias connecting a neuron to the paritcular output neuron.
            //
            // We have already calculated the first two, and we don't need to calculate the third term as it
            // evaluates to 1. (dZj/dBj)
            Matrix const & outputNeuronBiases = this->GetBiases();
            Matrix biasGradients = Matrix(1, this->GetNumNeurons());
            {
                for (u32 rIdx = 0; rIdx < biasGradients.GetHeight(); ++rIdx)
                {
                    f32 momentum = 0.0f;
                    if (m_PrevBiasGradients.GetCapacity() > 0)
                    {
                        momentum = m_PrevBiasGradients.GetElement(rIdx, 0) * momentumRate;
                    }

                    // Calculate the bias gradient (how sensitive the cost is to this particular bias)
                    biasGradients.GetElement(rIdx, 0) =
                        activationSensitivities.GetElement(rIdx, 0) * costSensitivities.GetElement(rIdx, 0) + momentum;
                }
            }

            // For each neuron in the previous layer, we need to compute how much its activation value
            // affects the total cost. 
            // 
            //        o
            //       /
            //      o
            //       \
            //        o
            //
            // i.e. The neuron on the left affects the cost, or error, in the two output neurons.
            // We will calculate this cost and then return the value for the next layer to use.
            Matrix prevLayerNeuronCostSensitivity = Matrix(1, prevLayer->GetNumNeurons());
            {
                Matrix const & outputNeuronWeights = this->GetWeights();

                for (u32 plvrIdx = 0; plvrIdx < prevLayer->GetValues().GetHeight(); ++plvrIdx)
                {
                    f32 sensitivity = 0.0f;
                    u32 numConnections = 0;

                    // Which neurons in the output layer are connected to this particular neuron in the previous layer?
                    for (u32 rIdx = 0; rIdx < outputNeuronWeights.GetHeight(); ++rIdx)
                    {
                        f32 const & weightValue = outputNeuronWeights.GetElement(rIdx, plvrIdx);
                        if (weightValue != 0.0f) // I.e. there is a connection! TODO: Set this to a global const?
                        {
                            sensitivity += 
                                weightValue * activationSensitivities.GetElement(rIdx, 0) * costSensitivities.GetElement(rIdx, 0);

                            numConnections += 1;
                        }
                    }

                    prevLayerNeuronCostSensitivity.GetElement(plvrIdx, 0) = sensitivity / numConnections;
                }
            }

            // Adjust the gradients by a learning rate
            weightGradients = Matrix::Multiply(weightGradients, learningRate);
            biasGradients = Matrix::Multiply(biasGradients, learningRate);

            m_PrevWeightGradients = weightGradients;
            m_PrevBiasGradients = biasGradients;

            // Subtract the weight & bias gradients from the weights & biases for the output layer
            m_Weights = Matrix::Subtract(m_Weights, weightGradients);
            m_Biases = Matrix::Subtract(m_Biases, biasGradients);

            // Return
            return prevLayerNeuronCostSensitivity;
        }
    }
}
