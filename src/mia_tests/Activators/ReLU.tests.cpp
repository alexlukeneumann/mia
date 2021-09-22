#include <CppUnitTest.h>

#include <Activators/ReLU.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace mia
{
    namespace tests
    {
        TEST_CLASS(ReLUTests)
        {
            static f32 constexpr c_Precision = 1e-3f;

        public:
            TEST_METHOD(ReturnsZero_ForAllNegativeValues)
            {
                Assert::AreEqual(0.0f, activators::ReLU(-1.0f));
                Assert::AreEqual(0.0f, activators::ReLU(-12.0f));
                Assert::AreEqual(0.0f, activators::ReLU(-123.0f));
                Assert::AreEqual(0.0f, activators::ReLU(-1234.0f));
                Assert::AreEqual(0.0f, activators::ReLU(-12345.0f));
            }

            TEST_METHOD(ReturnsZero_ForZero)
            {
                Assert::AreEqual(0.0f, activators::ReLU(0));
            }

            TEST_METHOD(ReturnsX_ForAllPositiveValues)
            {
                Assert::AreEqual(1.0f, activators::ReLU(1.0f));
                Assert::AreEqual(12.0f, activators::ReLU(12.0f));
                Assert::AreEqual(123.0f, activators::ReLU(123.0f));
                Assert::AreEqual(1234.0f, activators::ReLU(1234.0f));
                Assert::AreEqual(12345.0f, activators::ReLU(12345.0f));
            }
        };
    }
}
