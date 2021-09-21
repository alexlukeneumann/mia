#pragma once

#include "Maths/Matrix.h"

namespace mia
{
    namespace Tests
    {
        class LayerManipulator;
    }

    enum class LayerType : u8
    {
        Generic,
        Input
    };

    class Layer
    {
    public:
        Layer() = default;
        Layer(Layer const & other) = delete;
        Layer(Layer && other) = delete;
        virtual ~Layer() = default;

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
        // A matrix storing each neuron's weights.
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

        // A matrix storing the previous layer's neuron values multiplied by m_Weights structure.
        // This matrix is always expected to be one-dimensional in the height axis... i.e. Matrix(width: 1, height: n).
        // This allows for applying the summation, for a particular neuron, of the dot product of a previous neuron
        // and its weight easily... i.e. Matrix::Multiply(m_Weights, prevLayer.m_Values) 
        Matrix m_Values;

    protected:
        friend class Tests::LayerManipulator;
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

    inline void Layer::Execute(Layer const * prevLayer)
    {
        ASSERTMSG(nullptr != prevLayer, "prevLayer is not a valid ptr.");

        // Base implemention:
        // - Multiply the pre-filled m_Weights structure by the prevLayer's
        // already calculated m_Values structure.
        // - Add the m_Biases matrix to the m_Values structure
        //
        // The above two steps essentially summates the dot product of each neuron,
        // in the previous layer, connected to a particular neuron with its weight and
        // then adds the neuron's particular bias value to the final value for the neuron.
        m_Values = Matrix::Multiply(m_Weights, prevLayer->GetValues());
        m_Values = Matrix::Add(m_Values, m_Biases);
    }

    inline u32 Layer::GetNumNeurons() const
    {
        return static_cast<u32>(m_Values.GetCapacity());
    }
}
