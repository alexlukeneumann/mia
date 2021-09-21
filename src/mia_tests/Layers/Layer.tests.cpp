#include <CppUnitTest.h>

#include "Helpers/LayerManipulator.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace mia
{
    namespace Tests
    {
        class TestLayer : public Layer
        {
        public:
            virtual void Compile(u32 seedValue, Layer const * prevLayer) {}
        };

        TEST_CLASS(LayerTests)
        {
            static f32 constexpr c_Precision = 1e-3f;

        public:
            TEST_METHOD(BaseExecute_CalculatesTheCorrectSummationOfDotProducts_ForEachNeuron)
            {
                TestLayer prevLayer;
                TestLayer layer;

                // Initialise the previous layer's neuron values matrix
                Matrix & prevLayerValues = LayerManipulator::GetValuesMatrix(prevLayer);
                f32 values[] = {
                    5.0f,
                    6.0f,
                    7.0f
                };
                prevLayerValues = Matrix(1, 3, values);

                // Initialise the layer's neuron weights matrix.
                Matrix & layerWeights = LayerManipulator::GetWeightsMatrix(layer);
                f32 weights[] = {
                    1.5f, 0.5f, 2.0f,
                    2.5f, 1.0f, 3.0f
                };
                layerWeights = Matrix(3, 2, weights);

                // Execute the layer
                layer.Execute(&prevLayer);

                // Check the resulting values are as expected
                f32 const expectedValues[] = {
                    24.5f,  /* (1.5 * 5.0) + (0.5 * 6.0) + (2.0 * 7.0) */
                    39.5f   /* (2.5 * 5.0) + (1.0 * 6.0) + (3.0 * 7.0) */
                };

                Matrix const & calculatedValues = layer.GetValues();
                for (u64 rIdx = 0; rIdx < calculatedValues.GetHeight(); ++rIdx)
                {
                    Assert::IsTrue(abs(calculatedValues.GetElement(rIdx, 0) - expectedValues[rIdx]) < c_Precision);
                }
            }
        };
    }
}
