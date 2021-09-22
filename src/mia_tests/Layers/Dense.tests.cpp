#include <CppUnitTest.h>

#include <Layers/Dense.h>
#include "Helpers/LayerManipulator.h"
#include "Helpers/TestInputLayer.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace mia
{
    namespace tests
    {
        TEST_CLASS(DenseTests)
        {
            static u32 constexpr c_TestSeedValue = 0;

        public:
            TEST_METHOD(Compile_ReservesCorrectSpaceInValuesMatrix)
            {
                // Setup previous layer
                u32 const prevLayerNumNeurons = 10;
                TestInputLayer prevLayer(prevLayerNumNeurons);
                prevLayer.Compile(c_TestSeedValue, nullptr);

                // Setup this layer
                u32 const layerNumNeurons = 128;
                layers::Dense layer(layerNumNeurons);
                layer.Compile(c_TestSeedValue, &prevLayer);

                // Check values matrix is populated correctly
                Matrix const & valuesMatrix = layer.GetValues();
                u32 const expectedNumNeurons = layerNumNeurons;
                Assert::AreEqual(expectedNumNeurons, layer.GetNumNeurons());
                Assert::AreEqual(expectedNumNeurons, static_cast<u32>(valuesMatrix.GetCapacity()));
                Assert::AreEqual(static_cast<u32>(expectedNumNeurons), valuesMatrix.GetHeight());
                Assert::AreEqual(static_cast<u32>(1), valuesMatrix.GetWidth());
            }

            TEST_METHOD(Compile_ReservesCorrectSpaceInWeightsMatrix)
            {
                // Setup previous layer
                u32 const prevLayerNumNeurons = 10;
                TestInputLayer prevLayer(prevLayerNumNeurons);
                prevLayer.Compile(c_TestSeedValue, nullptr);

                // Setup this layer
                u32 const layerNumNeurons = 128;
                layers::Dense layer(layerNumNeurons);
                layer.Compile(c_TestSeedValue, &prevLayer);

                // Check weights matrix is populated correctly
                Matrix const & weightsMatrix = layer.GetWeights();
                u32 const expectedCapacity = prevLayerNumNeurons * layerNumNeurons;
                Assert::AreEqual(expectedCapacity, static_cast<u32>(weightsMatrix.GetCapacity()));
                Assert::AreEqual(static_cast<u32>(layerNumNeurons), weightsMatrix.GetHeight());
                Assert::AreEqual(static_cast<u32>(prevLayerNumNeurons), weightsMatrix.GetWidth());
            }

            TEST_METHOD(Compile_ReservesCorrectSpaceInBiasesMatrix)
            {
                // Setup previous layer
                u32 const prevLayerNumNeurons = 10;
                TestInputLayer prevLayer(prevLayerNumNeurons);
                prevLayer.Compile(c_TestSeedValue, nullptr);

                // Setup this layer
                u32 const layerNumNeurons = 128;
                layers::Dense layer(layerNumNeurons);
                layer.Compile(c_TestSeedValue, &prevLayer);

                // Check values matrix is populated correctly
                Matrix const & biasesMatrix = layer.GetBiases();
                u32 const expectedNumNeurons = layerNumNeurons;
                Assert::AreEqual(expectedNumNeurons, layer.GetNumNeurons());
                Assert::AreEqual(expectedNumNeurons, static_cast<u32>(biasesMatrix.GetCapacity()));
                Assert::AreEqual(static_cast<u32>(expectedNumNeurons), biasesMatrix.GetHeight());
                Assert::AreEqual(static_cast<u32>(1), biasesMatrix.GetWidth());
            }

            TEST_METHOD(Compile_SeedsWeightsMatrix)
            {
                // Setup previous layer
                u32 const prevLayerNumNeurons = 10;
                TestInputLayer prevLayer(prevLayerNumNeurons);
                prevLayer.Compile(c_TestSeedValue, nullptr);

                // Setup this layer
                u32 const layerNumNeurons = 128;
                layers::Dense layer(layerNumNeurons);
                layer.Compile(c_TestSeedValue, &prevLayer);

                // Check weights matrix is populated correctly
                Matrix const & weightsMatrix = layer.GetWeights();
                for (u32 rIdx = 0; rIdx < weightsMatrix.GetHeight(); ++rIdx)
                {
                    for (u32 cIdx = 0; cIdx < weightsMatrix.GetWidth(); ++cIdx)
                    {
                        Assert::AreNotEqual(0.0f, weightsMatrix.GetElement(rIdx, cIdx));
                    }
                }
            }

            TEST_METHOD(Compile_SeedsBiasesMatrix)
            {
                // Setup previous layer
                u32 const prevLayerNumNeurons = 10;
                TestInputLayer prevLayer(prevLayerNumNeurons);
                prevLayer.Compile(c_TestSeedValue, nullptr);

                // Setup this layer
                u32 const layerNumNeurons = 128;
                layers::Dense layer(layerNumNeurons);
                layer.Compile(c_TestSeedValue, &prevLayer);

                // Check biases matrix is populated correctly
                Matrix const & biasesMatrix = layer.GetBiases();
                for (u32 rIdx = 0; rIdx < biasesMatrix.GetHeight(); ++rIdx)
                {
                    for (u32 cIdx = 0; cIdx < biasesMatrix.GetWidth(); ++cIdx)
                    {
                        Assert::AreNotEqual(0.0f, biasesMatrix.GetElement(rIdx, cIdx));
                    }
                }
            }
        };
    }
}
