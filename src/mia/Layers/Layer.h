#pragma once

#include "Maths/Matrix.h"

namespace mia
{
    namespace Tests
    {
        class LayerManipulator;
    }

    class Layer
    {
    public:
        Layer() = default;
        Layer(Layer const & other) = delete;
        Layer(Layer && other) = delete;
        virtual ~Layer() = default;

        // Sets up the layer.
        virtual void Compile(Layer const * prevLayer) = 0;

        // Executes the current layers operation on the supplied previous layer and stores
        // the computed values within the m_Values matrix.
        virtual void Execute(Layer const * prevLayer);

        // Returns the number of neurons in the layer.
        // This is computed at construction time of the layer.
        u64 GetNumNeurons() const;

        // Returns the matrix representing the connections between this layer and the previous
        // layer. These connections are denoted by a weight value.
        Matrix const & GetWeights() const;
        // Returns the matrix representing the computed neuron values for this layer.
        Matrix const & GetValues() const;

    protected:
        // A matrix storing each neuron's weights.
        // A neuron in the m_Values structure stores which neurons, in the previous layer, connect to it
        // in a row-majored fashion.
        // 
        // i.e.
        // 
        //     m_Weights       Prev Layer Neuron Values
        //  ---------------           -------
        //  | 1.0 0.5 1.5 |           | 4.0 |
        //  | 0.5 1.0 2.0 |     x     | 2.4 |
        //  ---------------           | 1.3 |
        //                            -------
        // 
        // This layer represents a dense layer where every neuron in the previous layer is connected to each
        // neuron in this layer. Therefore, this layer has two neurons.
        // 
        // With this setup, the value of the second neuron would be:
        // N1 = (0.5 * 4.0) + (1.0 * 2.4) + (2.0 * 1.3)
        // as all three neurons in the previous layer are connected to the second neuron in this layer.
        Matrix m_Weights;

        // A matrix storing the previous layer's neuron values multiplied by m_Weights structure.
        // This matrix is always expected to be one-dimensional in height axis... i.e. Matrix(width: 1, height: n).
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

    inline Matrix const & Layer::GetValues() const
    {
        return m_Values;
    }

    inline void Layer::Execute(Layer const * prevLayer)
    {
        // Base implemention:
        // Just need to multiply the pre-filled m_Weights structure by the prevLayer's
        // already calculated m_Values structure.
        ASSERTMSG(nullptr != prevLayer, "prevLayer is not a valid ptr.");
        m_Values = Matrix::Multiply(m_Weights, prevLayer->GetValues());
    }

    inline u64 Layer::GetNumNeurons() const
    {
        return m_Values.GetCapacity();
    }
}
