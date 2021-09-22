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

        void Sequential::Train(NDArrayView<f32> const & inputData, std::initializer_list<f32> const & expectedOutput)
        {
            // Pass the input data into the first layer
            static_cast<layers::InputLayer *>(m_Layers[0])->SetInputData(inputData);

            // Execute the current model based on the new input
            ForwardPropagation();

            // TODO: Backpropagation
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
    }
}
