#pragma once

#include "Maths/Matrix.h"
#include "Activators/Activators.h"

namespace mia
{
    namespace tests
    {
        class LayerManipulator;
    }

    namespace layers
    {
        enum class LayerType : u8
        {
            Generic,
            Input
        };

        class Layer
        {
        public:
            Layer() = delete;
            Layer(Layer const & other) = delete;
            Layer(Layer && other) = delete;
            virtual ~Layer() = default;

            // Constructs a layer.
            Layer(activators::ActivatorType activatorType);

            // Sets up the layer.
            virtual void Compile(u32 seedValue, Layer const * prevLayer) = 0;

            // Executes the current layers operation on the supplied previous layer and stores
            // the computed values within the m_Values matrix.
            virtual void Execute(Layer const * prevLayer);

            // Returns the type of this layer.
            virtual LayerType GetType() const { return LayerType::Generic; }

            // Returns the number of neurons in the layer.
            // This is computed at construction time of the layer.
            u32 GetNumNeurons() const;

            // Returns the matrix representing the connections between this layer and the previous
            // layer. These connections are denoted by a weight value.
            Matrix const & GetWeights() const;
            // Returns the 1D matrix representing the bias value for each neuron in this layer.
            Matrix const & GetBiases() const;
            // Returns the 1D matrix representing the computed neuron values for this layer.
            Matrix const & GetValues() const;

        protected:
            friend class tests::LayerManipulator;

        protected:
            // A matrix storing each neuron connection pair's weights.
            // A neuron in the m_Values structure stores which neurons, in the previous layer, connect to it
            // in a row-majored fashion.
            // 
            // i.e.
            // 
            //     m_Weights       Prev Layer Neuron Values     This Layer Neuron Values
            //  ---------------           -------                       ------
            //  | 1.0 0.5 1.5 |           | 4.0 |                       | n0 |
            //  | 0.5 1.0 2.0 |     x     | 2.4 |           =           | n1 |
            //  ---------------           | 1.3 |                       ------
            //                            -------
            // 
            // This layer represents a dense layer where every neuron in the previous layer is connected to each
            // neuron in this layer. Therefore, this layer has two neurons.
            // 
            // With this setup, the value of the second neuron would be:
            // n1 = (0.5 * 4.0) + (1.0 * 2.4) + (2.0 * 1.3)
            // as all three neurons in the previous layer are connected to the second neuron in this layer.
            Matrix m_Weights;

            // A matrix storing the bias value for each neuron in this particular layer.
            // This matrix is always expected to be one-dimensional in the height axis... i.e. Matrix(width: 1, height: n).
            // This allows for easily adding each neuron's bias value the computed value for the neuron...
            // i.e. Matrix::Add(m_Values, m_Biases)
            Matrix m_Biases;

            // A matrix storing the previous layer's neuron values multiplied by m_Weights structure (+bias & activation).
            // This matrix is always expected to be one-dimensional in the height axis... i.e. Matrix(width: 1, height: n).
            // This allows for applying the summation, for a particular neuron, of the dot product of a previous neuron
            // and its weight easily... i.e. Matrix::Multiply(m_Weights, prevLayer.m_Values) 
            Matrix m_Values;

        private:
            // A matrix storing the previous layer's neuron values multiplied by m_Weights structure (+bias). This stores
            // the same values as m_Values does but minus the activation function running on each neuron.
            // We cache the neuron values prior to the activator being applied so that this value can be used within backpropagation
            Matrix m_ValuesPriorActivator;

            // An enum specifying which support activation function should be applied to every neuron's computed value
            // during the execution of the layer.
            activators::ActivatorType m_ActivatorType;
        };

        inline Matrix const & Layer::GetWeights() const
        {
            return m_Weights;
        }

        inline Matrix const & Layer::GetBiases() const
        {
            return m_Biases;
        }

        inline Matrix const & Layer::GetValues() const
        {
            return m_Values;
        }

        inline u32 Layer::GetNumNeurons() const
        {
            return static_cast<u32>(m_Values.GetCapacity());
        }
    }
}
