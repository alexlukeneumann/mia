#include <CppUnitTest.h>

#include "TestInputLayer.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace mia
{
    namespace tests
    {
        TEST_CLASS(TestInputLayerTests)
        {
        public:
            TEST_METHOD(Compile_ReservesCorrectSpaceInValuesMatrix)
            {
                u32 const numNeurons = 128;
                TestInputLayer layer(numNeurons);

                layer.Compile(0, nullptr);

                Matrix const & valuesMatrix = layer.GetValues();
                Matrix const & weightsMatrix = layer.GetWeights();
                Assert::AreEqual(numNeurons, valuesMatrix.GetHeight());
                Assert::AreEqual(static_cast<u32>(1), valuesMatrix.GetWidth());
            }

            TEST_METHOD(SetInputData_CorrectlyPopulatesTheValuesMatrix)
            {
                u32 const numNeurons = 5;
                TestInputLayer layer(numNeurons);

                f32 inputData[] = { 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };

                layer.Compile(0, nullptr);
                layer.SetInputData({{ numNeurons, inputData }});

                Matrix const & valuesMatrix = layer.GetValues();
                for (u32 rIdx = 0; rIdx < valuesMatrix.GetHeight(); ++rIdx)
                {
                    Assert::AreEqual(inputData[rIdx], valuesMatrix.GetElement(rIdx, 0));
                }
            }
        };
    }
}
