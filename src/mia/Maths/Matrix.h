#pragma once

#include "Common.h"

namespace mia
{
    // Represents a 2D array of f32 data of arbitary size (row-major).
    class Matrix final
    {
    public:
        Matrix();
        Matrix(Matrix const & other);
        Matrix(Matrix && other) noexcept;
        ~Matrix();

        // Constructs a matrix of size width x height using the supplied data (this data is copied)
        Matrix(u32 width, u32 height, f32 * data);
        // Constructs a matrix of size width x height (allocates memory with each element being 0.0f)
        Matrix(u32 width, u32 height);

        Matrix & operator = (Matrix const & other);
        Matrix & operator = (Matrix && other) noexcept;

        // Fills the matrix with random values between 0 & 1 based on the supplied seed 
        void Seed(u32 seed);

        // Copies the supplied data within the matrix in a row-majored fashion.
        void Copy(u32 rowIndex, u32 colIndex, f32 const * data, u32 length);

        // Returns the width of the matrix
        u32 GetWidth() const;
        // Returns the height of the matrix
        u32 GetHeight() const;
        // Returns the maximum capacity of the matrix (width * height)
        u64 GetCapacity() const;

        // Returns the element associated with the supplied row index & height index
        f32 & GetElement(u64 rowIndex, u64 colIndex);
        f32 const & GetElement(u64 rowIndex, u64 colIndex) const;

        // Prints the matrix to the standard input/output stream
        void Print() const;

        // Multiplies matrix a by matrix b and returns the result. Both a & b are expected to be
        // row-majored as the supplied matrix b will be transposed during the multiply for better
        // cache-miss performance.
        static Matrix Multiply(Matrix const & a, Matrix const & b);
        // Multiplies matrix a by the supplied scalar value and returns the result.
        static Matrix Multiply(Matrix const & a, f32 scalar);
        // Transposes the supplied matrix and returns the result
        static Matrix Transpose(Matrix const & m);
        // Adds matrix a & b together and returns the result. Both a & b are expected to be the
        // exact same dimensions.
        static Matrix Add(Matrix const & a, Matrix const & b);
        // Subtracts matrix b from a and returns the result. Both a & b are expected to be the
        // exact same dimensions.
        static Matrix Subtract(Matrix const & a, Matrix const & b);

        bool operator == (Matrix const & other) const;
        bool operator != (Matrix const & other) const;

    private:
        u32 m_Width;
        u32 m_Height;
        f32 * m_Data;
    };

    inline u32 Matrix::GetWidth() const
    {
        return m_Width;
    }

    inline u32 Matrix::GetHeight() const
    {
        return m_Height;
    }

    inline u64 Matrix::GetCapacity() const
    {
        return static_cast<u64>(m_Width) * static_cast<u64>(m_Height);
    }

    inline f32 & Matrix::GetElement(u64 rowIndex, u64 colIndex)
    {
        ASSERTMSG(rowIndex < m_Height, "rowIndex out of bounds!");
        ASSERTMSG(colIndex < m_Width, "colIndex out of bounds!");

        return m_Data[(m_Width * rowIndex) + colIndex];
    }

    inline f32 const & Matrix::GetElement(u64 rowIndex, u64 colIndex) const
    {
        ASSERTMSG(rowIndex < m_Height, "rowIndex out of bounds!");
        ASSERTMSG(colIndex < m_Width, "colIndex out of bounds!");

        return m_Data[(m_Width * rowIndex) + colIndex];
    }
}
