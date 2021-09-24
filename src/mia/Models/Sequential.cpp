#include "Sequential.h"

#include "Layers/Layer.h"
#include "Layers/InputLayer.h"

namespace mia
{
    namespace models
    {
        Sequential::Sequential(std::initializer_list<layers::Layer *> const & layers)
            : m_NumLayers(static_cast<u32>(layers.size()))
            , m_Layers()
        {
            ASSERTMSG(m_NumLayers <= c_MaxNumLayers, "Sequential Model only supports 256 sequential layers.");

            u32 layerIndex = 0;
            for (auto iter = layers.begin(); iter != layers.end(); ++iter)
            {
                m_Layers[layerIndex++] = *iter;
            }

            for (; layerIndex < c_MaxNumLayers; ++layerIndex)
            {
                m_Layers[layerIndex] = nullptr;
            }
        }

        Sequential::~Sequential()
        {
            for (u32 layerIndex = 0; layerIndex < c_MaxNumLayers; ++layerIndex)
            {
                if (nullptr != m_Layers[layerIndex])
                {
                    delete m_Layers[layerIndex];
                    m_Layers[layerIndex] = nullptr;
                    continue;
                }
            
                break;
            }
        }

        void Sequential::Compile(u32 seedValue)
        {
            ASSERTMSG(m_NumLayers > 0, "Sequential Model cannot have zero layers.");
            ASSERTMSG(layers::LayerType::Input == m_Layers[0]->GetType(), "Sequential Model's first layer isn't an input layer.");

            u32 layerIndex = 0;
            layers::Layer * layer = m_Layers[layerIndex];
            layers::Layer * prevLayer = nullptr;
            while (nullptr != layer)
            {
                layer->Compile(seedValue, prevLayer);
                prevLayer = layer;
                layer = m_Layers[++layerIndex];
            }
        }

        f32 Sequential::Train(NDArrayView<f32> const & inputData, std::initializer_list<f32> const & expectedOutput)
        {
            ASSERTMSG(expectedOutput.size() == m_Layers[m_NumLayers - 1]->GetNumNeurons(), "Number of expected output values differs from the number of output neurons.");

            // Pass the input data into the first layer
            static_cast<layers::InputLayer *>(m_Layers[0])->SetInputData(inputData);

            // Execute the current model based on the new input
            ForwardPropagation();

            // Evaluate how to change each weight & bias in order to minimise the error 
            // (difference in the computed value and expected value for each neuron).
            // TODO: Backpropagation is expensive to run every time, we could reduce the cost by
            // doing mini-batches.
            BackPropagation(expectedOutput);

            // Calculate MSE
            f32 meanSquareError = 0.0f;
            {
                Matrix const & outputValues = m_Layers[m_NumLayers - 1]->GetValues();

                u32 rIdx = 0;
                for (auto iter = expectedOutput.begin(); iter != expectedOutput.end(); ++iter)
                {
                    meanSquareError += pow((outputValues.GetElement(rIdx++, 0) - *iter), 2);
                }

                meanSquareError /= static_cast<f32>(expectedOutput.size());
            }

            return meanSquareError;
        }

        Matrix Sequential::Execute(NDArrayView<f32> const & inputData)
        {
            // Pass the input data into the first layer
            static_cast<layers::InputLayer *>(m_Layers[0])->SetInputData(inputData);

            // Execute the current model based on the new input
            ForwardPropagation();

            return m_Layers[m_NumLayers - 1]->GetValues();
        }

        void Sequential::ForwardPropagation()
        {
            // Call execute on each layer sequentially (this propagates foward through the model).
            u32 layerIndex = 0;
            layers::Layer * layer = m_Layers[layerIndex];
            layers::Layer * prevLayer = nullptr;
            while (nullptr != layer)
            {
                layer->Execute(prevLayer);

                prevLayer = layer;
                layer = m_Layers[++layerIndex];
            }
        }

        void Sequential::BackPropagation(std::initializer_list<f32> const & expectedOutput)
        {
            // Convert the expected output to a matrix
            Matrix expectedValues = Matrix(1, static_cast<u32>(expectedOutput.size()));
            {
                u32 rIdx = 0;
                for (auto iter = expectedOutput.begin(); iter != expectedOutput.end(); ++iter)
                {
                    expectedValues.GetElement(rIdx++, 0) = *iter;
                }
            }

            u32 layerIndex = m_NumLayers - 1;
            layers::Layer * layer = m_Layers[layerIndex];
            layers::Layer * prevLayer = m_Layers[layerIndex - 1];
            Matrix cost = std::move(expectedValues);
            bool isOutputLayer = true;
            while (nullptr != prevLayer)
            {
                cost = layer->Backpropagation(cost, isOutputLayer, prevLayer);

                prevLayer = (layerIndex >= 2) ? m_Layers[layerIndex - 2] : nullptr;
                layer = m_Layers[--layerIndex];
                isOutputLayer = false;
            }
        }
    }
}
