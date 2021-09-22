#include <CppUnitTest.h>

#include "Helpers/LayerManipulator.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace mia
{
    namespace tests
    {
        class TestNoActivatorLayer : public layers::Layer
        {
        public:
            TestNoActivatorLayer()
                : Layer(activators::ActivatorType::None)
            {
            }

            virtual void Compile(u32 seedValue, layers::Layer const * prevLayer) {}
        };

        class TestReLUActivatorLayer : public layers::Layer
        {
        public:
            TestReLUActivatorLayer()
                : Layer(activators::ActivatorType::ReLU)
            {
            }

            virtual void Compile(u32 seedValue, layers::Layer const * prevLayer) {}
        };

        TEST_CLASS(LayerTests)
        {
            static f32 constexpr c_Precision = 1e-3f;

        public:
            TEST_METHOD(BaseExecute_CalculatesTheCorrectSummationOfDotProducts_ForEachNeuron)
            {
                TestNoActivatorLayer prevLayer;
                TestNoActivatorLayer layer;

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

                // Initialise the layer's neuron biases matrix.
                Matrix & layerBiases = LayerManipulator::GetBiasesMatrix(layer);
                f32 biases[] = {
                    0.0f,
                    0.0f
                };
                layerBiases = Matrix(1, 2, biases);

                // Initialise the layer's neuron values matrix.
                Matrix & layerValues = LayerManipulator::GetValuesMatrix(layer);
                layerValues = Matrix(1, 2);

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

            TEST_METHOD(BaseExecute_CalculatesTheCorrectSummationOfDotProducts_AndAddsBias_ForEachNeuron)
            {
                TestNoActivatorLayer prevLayer;
                TestNoActivatorLayer layer;

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

                // Initialise the layer's neuron biases matrix.
                Matrix & layerBiases = LayerManipulator::GetBiasesMatrix(layer);
                f32 biases[] = {
                    20.0f,
                    10.0f
                };
                layerBiases = Matrix(1, 2, biases);

                // Initialise the layer's neuron values matrix.
                Matrix & layerValues = LayerManipulator::GetValuesMatrix(layer);
                layerValues = Matrix(1, 2);

                // Execute the layer
                layer.Execute(&prevLayer);

                // Check the resulting values are as expected
                f32 const expectedValues[] = {
                    44.5f,  /* (1.5 * 5.0) + (0.5 * 6.0) + (2.0 * 7.0) + 20.0f */
                    49.5f   /* (2.5 * 5.0) + (1.0 * 6.0) + (3.0 * 7.0) + 10.0f */
                };

                Matrix const & calculatedValues = layer.GetValues();
                for (u64 rIdx = 0; rIdx < calculatedValues.GetHeight(); ++rIdx)
                {
                    Assert::IsTrue(abs(calculatedValues.GetElement(rIdx, 0) - expectedValues[rIdx]) < c_Precision);
                }
            }

            TEST_METHOD(BaseExecute_CalculatesTheCorrectSummationOfDotProducts_AndCallsActivator_ForEachNeuron)
            {
                TestNoActivatorLayer prevLayer;
                TestReLUActivatorLayer layer;

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
                    -1.5f, -0.5f, -2.0f,
                    2.5f, 1.0f, 3.0f
                };
                layerWeights = Matrix(3, 2, weights);

                // Initialise the layer's neuron biases matrix.
                Matrix & layerBiases = LayerManipulator::GetBiasesMatrix(layer);
                f32 biases[] = {
                    0.0f,
                    0.0f
                };
                layerBiases = Matrix(1, 2, biases);

                // Initialise the layer's neuron values matrix.
                Matrix & layerValues = LayerManipulator::GetValuesMatrix(layer);
                layerValues = Matrix(1, 2);

                // Execute the layer
                layer.Execute(&prevLayer);

                // Check the resulting values are as expected
                f32 const expectedValues[] = {
                    0.0f,  /* ReLU[(-1.5 * 5.0) + (-0.5 * 6.0) + (-2.0 * 7.0) + 0.0f] = 0.0f */
                    39.5f   /* ReLU[(2.5 * 5.0) + (1.0 * 6.0) + (3.0 * 7.0) + 0.0f] = 49.5f */
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
