#include "Matrix.h"

namespace mia
{
    Matrix::Matrix()
        : m_Width(0)
        , m_Height(0)
        , m_Data(nullptr)
    {
    }

    Matrix::Matrix(u32 width, u32 height, f32 * data)
        : m_Width(width)
        , m_Height(height)
        , m_Data(nullptr)
    {
        u64 const capacity = GetCapacity();
        if (capacity > 0)
        {
            m_Data = new f32[capacity];
            memcpy(m_Data, data, capacity * sizeof(f32));
        }
    }

    Matrix::Matrix(u32 width, u32 height)
        : m_Width(width)
        , m_Height(height)
        , m_Data(nullptr)
    {
        u64 const capacity = GetCapacity();
        m_Data = new f32[capacity];
        for (u64 eIdx = 0; eIdx < capacity; ++eIdx)
        {
            m_Data[eIdx] = 0.0f;
        }
    }

    Matrix::Matrix(Matrix const & other)
        : m_Width(0)
        , m_Height(0)
        , m_Data(nullptr)
    {
        *this = other;
    }

    Matrix::Matrix(Matrix && other) noexcept
        : m_Width(0)
        , m_Height(0)
        , m_Data(nullptr)
    {
        *this = std::move(other);
    }

    Matrix::~Matrix()
    {
        if (nullptr != m_Data)
        {
            delete[] m_Data;
        }
    }

    Matrix & Matrix::operator = (Matrix const & other)
    {
        m_Width = other.m_Width;
        m_Height = other.m_Height;
        m_Data = nullptr;

        u64 const capacity = GetCapacity();
        if (capacity > 0)
        {
            m_Data = new f32[capacity];
            memcpy(m_Data, other.m_Data, capacity * sizeof(f32));
        }

        return *this;
    }

    Matrix & Matrix::operator = (Matrix && other) noexcept
    {
        m_Width = other.m_Width;
        m_Height = other.m_Height;
        m_Data = other.m_Data;

        other.m_Width = 0;
        other.m_Height = 0;
        other.m_Data = nullptr;

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

    void Matrix::Copy(u32 rowIndex, u32 colIndex, f32 const * data, u32 length)
    {
        ASSERTMSG(rowIndex < m_Width, "rowIndex is out of bounds.");
        ASSERTMSG(colIndex < m_Height, "colIndex is out of bounds.");
        
        if (m_Width == 1)
        {
            // If the matrix is one dimensional (in height) then we can treat it as one dimensional in width
            ASSERTMSG(colIndex + length <= m_Height, "length of the supplied data is out of bounds.");

            u32 tmp = colIndex;
            colIndex = rowIndex;
            rowIndex = tmp;
        }
        else
        {
            ASSERTMSG(colIndex + length <= m_Width, "length of the supplied data is out of bounds.");
        }

        f32 * dst = &GetElement(rowIndex, colIndex);
        memcpy(dst, data, length * sizeof(f32));
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

        for (u32 cIdx = 0; cIdx < result.GetWidth(); ++cIdx)
        {
            for (u32 rIdx = 0; rIdx < result.GetHeight(); ++rIdx)
            {
                u32 const aOffset = a.GetWidth() * rIdx;
                u32 const bOffset = bTransposed.GetWidth() * cIdx;
                u32 const resultOffset = result.GetWidth() * rIdx;

                for (u32 i = 0; i < a.GetWidth(); ++i)
                {
                    f32 const & aVal = a.m_Data[aOffset + i];
                    f32 const & bVal = bTransposed.m_Data[bOffset + i];

                    result.m_Data[resultOffset + cIdx] += aVal * bVal;
                }
            }
        }

        return result;
    }

    Matrix Matrix::Transpose(Matrix const & m)
    {
        Matrix result;

        if (m.GetHeight() == 1 || m.GetWidth() == 1)
        {
            // If m is a 1D matrix we will just make a simple copy of the matrix with
            // the height & width values swapped.
            result = Matrix(m.GetHeight(), m.GetWidth(), m.m_Data);
        }
        else
        {
            // If m is a 2D matrix, we will need to perform the transpose operation 
            // on the matrix's data.
            result = Matrix(m.GetHeight(), m.GetWidth());

            for (u32 rIdx = 0; rIdx < m.GetHeight(); ++rIdx)
            {
                for (u32 cIdx = 0; cIdx < m.GetWidth(); ++cIdx)
                {
                    result.m_Data[(result.GetWidth() * cIdx) + rIdx] = m.m_Data[(m.GetWidth() * rIdx) + cIdx];
                }
            }
        }

        return result;
    }

    bool Matrix::operator == (Matrix const & other) const
    {
        if ((GetWidth() != other.GetWidth()) || (GetHeight() != other.GetHeight()))
        {
            return false;
        }

        for (u32 rIdx = 0; rIdx < GetHeight(); ++rIdx)
        {
            for (u32 cIdx = 0; cIdx < GetWidth(); ++cIdx)
            {
                if (GetElement(rIdx, cIdx) != other.GetElement(rIdx, cIdx))
                {
                    return false;
                }
            }
        }

        return true;
    }

    bool Matrix::operator != (Matrix const & other) const
    {
        return !(*this == other);
    }
}
