#include <CppUnitTest.h>

#include "Maths/Matrix.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace mia
{
    TEST_CLASS(MatrixTests)
    {
        static f32 constexpr c_Precision = 1e-3f;

    public:
        TEST_METHOD(CanConstruct_AZeroFilledMatrix)
        {
            u32 const width = 3;
            u32 const height = 2;

            Matrix m(width, height);

            for (u32 rIdx = 0; rIdx < width; ++rIdx)
            {
                for (u32 hIdx = 0; hIdx < height; ++hIdx)
                {
                    Assert::AreEqual(0.0f, m.GetElement(rIdx, hIdx));
                }
            }
        }

        TEST_METHOD(CanConstruct_AMatrixFromAnExisting1DArray)
        {
            u32 const width = 2;
            u32 const height = 3;

            f32 array[] = {
                1.0f, 2.0f,
                4.2f, 3.0f,
                1.1f, 2.4f
            };

            Matrix m(width, height, array);

            for (u32 rIdx = 0; rIdx < width; ++rIdx)
            {
                for (u32 hIdx = 0; hIdx < height; ++hIdx)
                {
                    Assert::AreEqual(array[(width * hIdx) + rIdx], m.GetElement(rIdx, hIdx));
                }
            }
        }

        TEST_METHOD(CanCopyConstruct)
        {
            u32 const width = 2;
            u32 const height = 2;

            f32 array[] = {
                1.0f, 2.0f,
                4.2f, 3.0f
            };

            Matrix m(width, height, array);
            Matrix mCopy(m);

            Assert::AreEqual(m.GetWidth(), mCopy.GetWidth());
            Assert::AreEqual(m.GetHeight(), mCopy.GetHeight());

            for (u32 rIdx = 0; rIdx < width; ++rIdx)
            {
                for (u32 hIdx = 0; hIdx < height; ++hIdx)
                {
                    Assert::AreEqual(array[(width * hIdx) + rIdx], mCopy.GetElement(rIdx, hIdx));
                }
            }

            Assert::AreNotEqual(&mCopy.GetElement(0, 0), &m.GetElement(0, 0));
        }

        TEST_METHOD(CanMoveConstruct)
        {
            u32 const width = 2;
            u32 const height = 2;

            f32 array[] = {
                1.0f, 2.0f,
                4.2f, 3.0f
            };

            Matrix m(width, height, array);
            Matrix mMove(std::move(m));

            Assert::AreEqual(width, mMove.GetWidth());
            Assert::AreEqual(height, mMove.GetHeight());
            Assert::AreEqual(static_cast<u32>(0), m.GetWidth());
            Assert::AreEqual(static_cast<u32>(0), m.GetHeight());

            for (u32 rIdx = 0; rIdx < width; ++rIdx)
            {
                for (u32 hIdx = 0; hIdx < height; ++hIdx)
                {
                    Assert::AreEqual(array[(width * hIdx) + rIdx], mMove.GetElement(rIdx, hIdx));
                }
            }
        }

        TEST_METHOD(CanAssignElementValues)
        {
            u32 const width = 3;
            u32 const height = 2;

            Matrix m(width, height);

            for (u32 rIdx = 0; rIdx < width; ++rIdx)
            {
                for (u32 hIdx = 0; hIdx < height; ++hIdx)
                {
                    Assert::AreEqual(0.0f, m.GetElement(rIdx, hIdx));
                }
            }

            f32 const val01 = 2.4f;
            f32 const val10 = 3.2f;

            m.GetElement(0, 1) = 2.4f;
            m.GetElement(1, 0) = 3.2f;

            Assert::AreEqual(val01, m.GetElement(0, 1));
            Assert::AreEqual(val10, m.GetElement(1, 0));
        }

        TEST_METHOD(CanMultiply_TwoMatrices_SameDimensions)
        {
            u32 const width = 3;
            u32 const height = 3;

            f32 aValues[] = {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f
            };

            f32 bValues[] = {
                3.0f, 4.0f, 5.0f,
                1.0f, 2.0f, 3.0f,
                1.5f, 2.5f, 4.5f
            };

            f32 expected[] = {
                9.5f, 15.5f, 24.5f,
                26.0f, 41.0f, 62.0f,
                42.5f, 66.5f, 99.5f
            };

            Matrix a(width, height, aValues);
            Matrix b(width, height, bValues);

            Matrix result = Matrix::Multiply(a, b);

            for (u32 rIdx = 0; rIdx < width; ++rIdx)
            {
                for (u32 hIdx = 0; hIdx < height; ++hIdx)
                {
                    f32 const & expectedValue = expected[(width * hIdx) + rIdx];
                    f32 const & calculatedValue = result.GetElement(rIdx, hIdx);

                    Assert::IsTrue(abs(expectedValue - calculatedValue) < c_Precision);
                }
            }
        }

        TEST_METHOD(CanMultiply_TwoMatrices_DifferentDimensions)
        {
            u32 const aWidth = 3;
            u32 const aHeight = 3;
            u32 const bWidth = 2;
            u32 const bHeight = 3;

            f32 aValues[] = {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f
            };

            f32 bValues[] = {
                3.0f, 4.0f,
                1.0f, 2.0f,
                1.5f, 2.5f,
            };

            f32 expected[] = {
                9.5f, 15.5f, 
                26.0f, 41.0f,
                42.5f, 66.5f,
            };

            Matrix a(aWidth, aHeight, aValues);
            Matrix b(bWidth, bHeight, bValues);

            Matrix result = Matrix::Multiply(a, b);

            u32 const resultWidth = 2;
            u32 const resultHeight = 3;

            Assert::AreEqual(resultWidth, result.GetWidth());
            Assert::AreEqual(resultHeight, result.GetHeight());

            for (u32 rIdx = 0; rIdx < resultWidth; ++rIdx)
            {
                for (u32 hIdx = 0; hIdx < resultHeight; ++hIdx)
                {
                    f32 const & expectedValue = expected[(resultWidth * hIdx) + rIdx];
                    f32 const & calculatedValue = result.GetElement(rIdx, hIdx);

                    Assert::IsTrue(abs(expectedValue - calculatedValue) < c_Precision);
                }
            }
        }

        TEST_METHOD(CanTranspose_SameWidthAndHeight)
        {
            u32 const width = 3;
            u32 const height = 3;

            f32 values[] = {
                3.0f, 4.0f, 5.0f,
                1.0f, 2.0f, 3.0f,
                1.5f, 2.5f, 4.5f
            };

            f32 expected[] = {
                3.0f, 1.0f, 1.5f,
                4.0f, 2.0f, 2.5f,
                5.0f, 3.0f, 4.5f
            };

            Matrix m(width, height, values);
            Matrix mTransposed = Matrix::Transpose(m);

            Assert::AreEqual(width, mTransposed.GetWidth());
            Assert::AreEqual(height, mTransposed.GetHeight());

            for (u32 rIdx = 0; rIdx < mTransposed.GetWidth(); ++rIdx)
            {
                for (u32 hIdx = 0; hIdx < mTransposed.GetHeight(); ++hIdx)
                {
                    Assert::AreEqual(expected[(mTransposed.GetWidth() * hIdx) + rIdx], mTransposed.GetElement(rIdx, hIdx));
                }
            }
        }

        TEST_METHOD(CanTranspose_DifferentWidthAndHeight)
        {
            u32 const width = 3;
            u32 const height = 4;

            f32 values[] = {
                3.0f, 4.0f, 5.0f,
                1.0f, 2.0f, 3.0f,
                1.5f, 2.5f, 4.5f,
                2.0f, 7.0f, 6.0f
            };

            f32 expected[] = {
                3.0f, 1.0f, 1.5f, 2.0f,
                4.0f, 2.0f, 2.5f, 7.0f,
                5.0f, 3.0f, 4.5f, 6.0f
            };

            Matrix m(width, height, values);
            Matrix mTransposed = Matrix::Transpose(m);

            Assert::AreEqual(height, mTransposed.GetWidth());
            Assert::AreEqual(width, mTransposed.GetHeight());

            for (u32 rIdx = 0; rIdx < mTransposed.GetWidth(); ++rIdx)
            {
                for (u32 hIdx = 0; hIdx < mTransposed.GetHeight(); ++hIdx)
                {
                    Assert::AreEqual(expected[(mTransposed.GetWidth() * hIdx) + rIdx], mTransposed.GetElement(rIdx, hIdx));
                }
            }
        }
    };
}
