#pragma once

#include "Common.h"

namespace mia
{
    typedef u32 DimensionLength;

    template <class Type>
    struct NDArrayViewElement
    {
        DimensionLength length = 0;
        Type * data = nullptr;
    };

    // A simple class to describe a multi-dimensional dataset. This
    // class does not copy any of the dimensional data, it is merely a
    // "view" into an existing dataset.
    template <class Type>
    class NDArrayView
    {
    public:
        NDArrayView();
        NDArrayView(NDArrayView<Type> const & other);
        ~NDArrayView();

        // Constructs a NDArrayView with the supplied number of dimensions and an array of NDArrayViewElements of the same length
        NDArrayView(u32 numDimensions, NDArrayViewElement<Type> const * dimensions);

        NDArrayView<Type> & operator = (NDArrayView<Type> const & other);

        // Returns the number of dimensions this view covers
        u32 GetNumDimensions() const;
        // Returns the particular view element for a given dimension
        NDArrayViewElement<Type> const & GetDimension(u32 dimensionIndex) const;

    private:
        u32 m_NumDimensions;
        NDArrayViewElement<Type> * m_Dimensions;
    };

    template <class Type>
    NDArrayView<Type>::NDArrayView()
        : m_NumDimensions(0)
        , m_Dimensions(nullptr)
    {
    }

    template <class Type>
    NDArrayView<Type>::NDArrayView(u32 numDimensions, NDArrayViewElement<Type> const * dimensions)
        : m_NumDimensions(numDimensions)
        , m_Dimensions(nullptr)
    {
        if (numDimensions > 0)
        {
            m_Dimensions = new NDArrayViewElement<Type>[numDimensions];
            memcpy(m_Dimensions, dimensions, numDimensions * sizeof(NDArrayViewElement<Type>));
        }
    }

    template <class Type>
    NDArrayView<Type>::NDArrayView(NDArrayView<Type> const & other)
        : m_NumDimensions(0)
        , m_Dimensions(nullptr)
    {
        *this = other;
    }

    template <class Type>
    NDArrayView<Type>::~NDArrayView()
    {
        if (nullptr != m_Dimensions)
        {
            delete[] m_Dimensions;
        }
    }

    template <class Type>
    NDArrayView<Type> & NDArrayView<Type>::operator = (NDArrayView<Type> const & other)
    {
        m_NumDimensions = other.m_NumDimensions;
        m_Dimensions = nullptr;

        if (m_NumDimensions > 0)
        {
            m_Dimensions = new NDArrayViewElement<Type>[m_NumDimensions];
            memcpy(m_Dimensions, other.m_Dimensions, m_NumDimensions * sizeof(NDArrayViewElement<Type>));
        }

        return *this;
    }

    template <class Type>
    inline u32 NDArrayView<Type>::GetNumDimensions() const
    {
        return m_NumDimensions;
    }

    template <class Type>
    inline NDArrayViewElement<Type> const & NDArrayView<Type>::GetDimension(u32 dimensionIndex) const
    {
        ASSERTMSG(dimensionIndex < m_NumDimensions, "dimensionIndex is out of bounds.");
        return m_Dimensions[dimensionIndex];
    }
}
