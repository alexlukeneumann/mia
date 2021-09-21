#include <CppUnitTest.h>

#include <Layers/Flatten.h>
#include "Helpers/LayerManipulator.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace mia
{
    namespace Tests
    {
        TEST_CLASS(FlattenTests)
        {
        public:
            TEST_METHOD(Compile_ReservesCorrectSpaceInValuesMatrix)
            {
                u32 const numDimensions = 3;
                Flatten layer({3, 3, 2});

                Assert::AreEqual(static_cast<u64>(0), layer.GetNumNeurons());

                layer.Compile(nullptr);

                Matrix const & valuesMatrix = layer.GetValues();
                u64 expectedNumNeurons = 3 * 3 * 2;
                Assert::AreEqual(expectedNumNeurons, valuesMatrix.GetCapacity());
                Assert::AreEqual(static_cast<u32>(expectedNumNeurons), valuesMatrix.GetHeight());
                Assert::AreEqual(static_cast<u32>(1), valuesMatrix.GetWidth());
                Assert::AreEqual(expectedNumNeurons, layer.GetNumNeurons());
            }

            TEST_METHOD(SetInputData_PopulatesValuesMatrixCorrectly)
            {
                // Create layer
                u32 const numDimensions = 3;
                Flatten layer({3, 3, 2});

                Assert::AreEqual(static_cast<u64>(0), layer.GetNumNeurons());

                layer.Compile(nullptr);

                // Create input data
                f32 xData[] = { 2.0f, 3.0f, 4.0f };
                f32 yData[] = { 3.0f, 4.0f, 5.0f };
                f32 zData[] = { 5.0f, 6.0f };
                
                NDArrayViewElement<f32> viewElements[] = {
                    { 3, xData },
                    { 3, yData },
                    { 2, zData }
                };

                NDArrayView<f32> arrayView(numDimensions, viewElements);

                // Populate layer with input data
                layer.SetInputData(arrayView);

                // Check values matrix is populated correctly
                Matrix const & valuesMatrix = layer.GetValues();
                u32 matrixOffset = 0;
                for (u32 dIdx = 0; dIdx < numDimensions; ++dIdx)
                {
                    NDArrayViewElement<f32> viewElement = viewElements[dIdx];

                    for (u32 eIdx = 0; eIdx < viewElement.length; ++eIdx)
                    {
                        Assert::AreEqual(valuesMatrix.GetElement(0, matrixOffset), viewElement.data[eIdx]);
                        matrixOffset++;
                    }
                }
            }

            TEST_METHOD(Execute_DoesntDoAnything)
            {
                // Create layer
                u32 const numDimensions = 3;
                Flatten layer({3, 3, 2});

                Assert::AreEqual(static_cast<u64>(0), layer.GetNumNeurons());

                layer.Compile(nullptr);

                // Check execute doesn't touch values matrix
                Matrix prevValues = layer.GetValues();
                layer.Execute(nullptr);
                Assert::IsTrue(prevValues == layer.GetValues());
            }
        };
    }
}
