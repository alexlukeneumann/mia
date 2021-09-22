#include <CppUnitTest.h>

#include <Core/NDArrayView.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace mia
{
    namespace tests
    {
        TEST_CLASS(NDArrayViewTests)
        {
        public:
            TEST_METHOD(CanConstruct_Default)
            {
                NDArrayView<f32> arrayView;
                Assert::AreEqual(static_cast<u32>(0), arrayView.GetNumDimensions());
            }

            TEST_METHOD(CanConstruct_WithNDimensionalData)
            {
                f32 xData[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
                f32 yData[] = { 2.0f, 1.0f, 4.0f };
                f32 zData[] = { 3.0f, 1.0f, 4.0f, 5.0f };

                NDArrayViewElement<f32> viewElements[] = {
                    { LENGTHOF(xData), xData },
                    { LENGTHOF(yData), yData },
                    { LENGTHOF(zData), zData }
                };
                NDArrayView<f32> arrayView(LENGTHOF(viewElements), viewElements);

                Assert::AreEqual(static_cast<u32>(LENGTHOF(viewElements)), arrayView.GetNumDimensions());
                Assert::IsTrue(viewElements != &arrayView.GetDimension(0));

                for (u32 dIdx = 0; dIdx < LENGTHOF(viewElements); ++dIdx)
                {
                    NDArrayViewElement<f32> const & storedViewElement = arrayView.GetDimension(dIdx);

                    Assert::AreEqual(viewElements[dIdx].length, storedViewElement.length);
                    Assert::AreEqual(viewElements[dIdx].data, storedViewElement.data);
                }
            }

            TEST_METHOD(CanConstruct_WithInitializerList)
            {
                f32 xData[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
                f32 yData[] = { 2.0f, 1.0f, 4.0f };
                f32 zData[] = { 3.0f, 1.0f, 4.0f, 5.0f };

                u32 const numDimensions = 3;
                NDArrayView<f32> arrayView({
                    { LENGTHOF(xData), xData },
                    { LENGTHOF(yData), yData },
                    { LENGTHOF(zData), zData }
                });

                Assert::AreEqual(numDimensions, arrayView.GetNumDimensions());

                Assert::AreEqual(static_cast<u32>(LENGTHOF(xData)), arrayView.GetDimension(0).length);
                Assert::AreEqual(static_cast<u32>(LENGTHOF(yData)), arrayView.GetDimension(1).length);
                Assert::AreEqual(static_cast<u32>(LENGTHOF(zData)), arrayView.GetDimension(2).length);

                Assert::AreEqual(&xData[0], arrayView.GetDimension(0).data);
                Assert::AreEqual(&yData[0], arrayView.GetDimension(1).data);
                Assert::AreEqual(&zData[0], arrayView.GetDimension(2).data);
            }

            TEST_METHOD(CanCopyConstruct)
            {
                f32 xData[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
                f32 yData[] = { 2.0f, 1.0f, 4.0f };
                f32 zData[] = { 3.0f, 1.0f, 4.0f, 5.0f };

                NDArrayViewElement<f32> viewElements[] = {
                    { LENGTHOF(xData), xData },
                    { LENGTHOF(yData), yData },
                    { LENGTHOF(zData), zData }
                };
                NDArrayView<f32> arrayView(LENGTHOF(viewElements), viewElements);
                NDArrayView<f32> copyArrayView(arrayView);

                Assert::AreEqual(arrayView.GetNumDimensions(), copyArrayView.GetNumDimensions());
                Assert::IsTrue(&arrayView.GetDimension(0) != &copyArrayView.GetDimension(0));
            
                for (u32 dIdx = 0; dIdx < LENGTHOF(viewElements); ++dIdx)
                {
                    NDArrayViewElement<f32> const & storedViewElement = arrayView.GetDimension(dIdx);
                    NDArrayViewElement<f32> const & copyStoredViewElement = copyArrayView.GetDimension(dIdx);

                    Assert::AreEqual(storedViewElement.length, copyStoredViewElement.length);
                    Assert::AreEqual(storedViewElement.data, copyStoredViewElement.data);
                }
            }
        };
    }
}
