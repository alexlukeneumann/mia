#include <CppUnitTest.h>

#include <Activators/Sigmoid.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace mia
{
    namespace Tests
    {
        TEST_CLASS(SigmoidTests)
        {
            static f32 constexpr c_Precision = 1e-3f;

        public:
            TEST_METHOD(eturnsOriginalValue_WhenApplyingTheInverse)
            {
                f32 const originalValue = 0.0f;
                f32 const sVal = activators::Sigmoid(0.0f);

                Assert::AreNotEqual(sVal, originalValue);

                f32 const iVal = log(sVal / (1 - sVal));

                Assert::AreEqual(originalValue, iVal);
            }

            TEST_METHOD(ReturnsOne_ForLargePositiveValues)
            {
                Assert::AreEqual(1.0f, activators::Sigmoid(100.0f));
            }

            TEST_METHOD(ReturnsZero_ForLargeNegativeValues)
            {
                Assert::AreEqual(0.0f, activators::Sigmoid(-100.0f));
            }
        };
    }
}
