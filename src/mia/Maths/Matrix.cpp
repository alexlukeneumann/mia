#include "Matrix.h"

namespace mia
{
    Matrix::Matrix(u32 width, u32 height, f32 * data)
        : m_Width(width)
        , m_Height(height)
        , m_Data(data)
        , m_ManagesMemory(false)
    {
    }

    Matrix::Matrix(u32 width, u32 height)
        : m_Width(width)
        , m_Height(height)
        , m_Data(nullptr)
        , m_ManagesMemory(false)
    {
        u64 const capacity = GetCapacity();
        m_Data = new f32[capacity];
        for (u64 eIdx = 0; eIdx < capacity; ++eIdx)
        {
            m_Data[eIdx] = 0.0f;
        }
        m_ManagesMemory = true;
    }

    Matrix::Matrix(Matrix const & other)
        : m_Width(0)
        , m_Height(0)
        , m_Data(nullptr)
        , m_ManagesMemory(false)
    {
        *this = other;
    }

    Matrix::Matrix(Matrix && other) noexcept
        : m_Width(0)
        , m_Height(0)
        , m_Data(nullptr)
        , m_ManagesMemory(false)
    {
        *this = std::move(other);
    }

    Matrix::~Matrix()
    {
        if (m_ManagesMemory && (nullptr != m_Data))
        {
            delete m_Data;
            m_Data = nullptr;
            m_ManagesMemory = false;
        }
    }

    Matrix & Matrix::operator = (Matrix const & other)
    {
        m_Width = other.m_Width;
        m_Height = other.m_Height;
        m_Data = nullptr;
        m_ManagesMemory = false;

        u64 const capacity = GetCapacity();
        if (capacity > 0)
        {
            m_Data = new f32[capacity];
            for (u64 eIdx = 0; eIdx < capacity; ++eIdx)
            {
                m_Data[eIdx] = other.m_Data[eIdx];
            }
            m_ManagesMemory = true;
        }

        return *this;
    }

    Matrix & Matrix::operator = (Matrix && other) noexcept
    {
        m_Width = other.m_Width;
        m_Height = other.m_Height;
        m_Data = other.m_Data;
        m_ManagesMemory = other.m_ManagesMemory;

        other.m_Width = 0;
        other.m_Height = 0;
        other.m_Data = nullptr;
        other.m_ManagesMemory = false;

        return *this;
    }

    void Matrix::Seed(u32 seed)
    {
        ASSERTMSG(nullptr != m_Data, "Failed to seed matrix.");

        srand(seed);
        u64 const capacity = GetCapacity();
        for (u64 eIdx = 0; eIdx < capacity; ++eIdx)
        {
            m_Data[eIdx] = static_cast<f32>(rand()) / static_cast<f32>(RAND_MAX);
        }
    }

    f32 & Matrix::GetElement(u64 rowIndex, u64 heightIndex)
    {
        ASSERTMSG(rowIndex < m_Width, "rowIndex out of bounds!");
        ASSERTMSG(heightIndex < m_Height, "heightIndex out of bounds!");

        return m_Data[(m_Width * heightIndex) + rowIndex];
    }

    void Matrix::Print() const
    {
        u64 const capacity = GetCapacity();
        for (u64 eIdx = 0; eIdx < capacity; ++eIdx)
        {
            char tmp[32];
            sprintf_s(tmp, "%.4f", m_Data[eIdx]);

            std::cout << tmp;

            if ((eIdx + 1) % GetWidth() == 0)
            {
                std::cout << "\n";
            }
            else
            {
                std::cout << "\t";
            }
        }

        std::cout << std::endl;
    }

    Matrix Matrix::Multiply(Matrix const & a, Matrix const & b)
    {
        ASSERTMSG((nullptr != a.m_Data) && (nullptr != b.m_Data), "Failed to multipy matrix by another. Matrices aren't valid.");
        ASSERTMSG(a.GetWidth() == b.GetHeight(), "Impossible matrix multiplication. Incompatible dimensions.");

        /*
            Performance Note:
            To improve performance with large matrices, things to implement:
                - Multithreading (either on CPU or GPU)
                - SIMD operations
        */

        Matrix result(b.GetWidth(), a.GetHeight());

        // Transpose b to reduce cache misses on large matrices
        Matrix bTransposed = Matrix::Transpose(b);

        for (u32 rIdx = 0; rIdx < result.GetWidth(); ++rIdx)
        {
            for (u32 hIdx = 0; hIdx < result.GetHeight(); ++hIdx)
            {
                for (u32 i = 0; i < a.GetWidth(); ++i)
                {
                    f32 const & aVal = a.m_Data[(a.GetWidth() * hIdx) + i];
                    f32 const & bVal = bTransposed.m_Data[(bTransposed.GetWidth() * rIdx) + i];

                    result.m_Data[(result.GetWidth() * hIdx) + rIdx] += aVal * bVal;
                }
            }
        }

        return result;
    }

    Matrix Matrix::Transpose(Matrix const & m)
    {
        Matrix result(m.GetHeight(), m.GetWidth());

        for (u32 hIdx = 0; hIdx < m.GetHeight(); ++hIdx)
        {
            for (u32 rIdx = 0; rIdx < m.GetWidth(); ++rIdx)
            {
                result.m_Data[(result.GetWidth() * rIdx) + hIdx] = m.m_Data[(m.GetWidth() * hIdx) + rIdx];
            }
        }

        return result;
    }
}
