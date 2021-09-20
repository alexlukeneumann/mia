#include "Flatten.h"

namespace mia
{
    Flatten::Flatten()
        : m_InputNumDimensions(0)
        , m_InputDimensionLengths(nullptr)
    {
    }

    Flatten::~Flatten()
    {
        if (nullptr != m_InputDimensionLengths)
        {
            delete[] m_InputDimensionLengths;
        }
    }

    Flatten::Flatten(u32 inputNumDimensions, DimensionLength * inputDimensionLengths)
        : m_InputNumDimensions(inputNumDimensions)
        , m_InputDimensionLengths(nullptr)
    {
        ASSERTMSG(m_InputNumDimensions != 0, "Cannot create a Flatten layer with expected data containing 0 dimensions.");
        m_InputDimensionLengths = new u32[inputNumDimensions];
        memcpy(m_InputDimensionLengths, inputDimensionLengths, m_InputNumDimensions * sizeof(u32));
    }

    void Flatten::Compile(Layer const * prevLayer)
    {
        ASSERTMSG(prevLayer == nullptr, "Flatten layer cannot be used superceding another 1D layer.");

        // Calculate required capacity
        u32 capacity = m_InputDimensionLengths[0];
        for (u32 dIdx = 1; dIdx < m_InputNumDimensions; ++dIdx)
        {
            capacity *= m_InputDimensionLengths[dIdx];
        }
        ASSERTMSG(capacity > 0, "Calculated capacity for the Flatten layer is zero.");

        // Initialise the m_Values matrix
        m_Values = Matrix(1, capacity);
    }

    void Flatten::SetInputData(NDArrayView<f32> const & inputData)
    {
        // Check the supplied inputData matches the expected shape
        ASSERTMSG(GetNumNeurons() > 0, "Flatten layer's compile hasn't been called.");
        ASSERTMSG(m_InputNumDimensions == inputData.GetNumDimensions(), "Supplied input data to the flatten layer doesn't match the expected input shape.");

        for (u32 dIdx = 0; dIdx < m_InputNumDimensions; ++dIdx)
        {
            u32 const expectedDimensionLength = m_InputDimensionLengths[dIdx];
            u32 const suppliedDimensionLength = inputData.GetDimension(dIdx).length;

            ASSERTMSG(expectedDimensionLength == suppliedDimensionLength, "Supplied input data to the flatten layer doesn't match the expected input shape.");
        }

        // Copy data into the m_Values matrix
        u32 matrixOffset = 0;
        for (u32 dIdx = 0; dIdx < inputData.GetNumDimensions(); ++dIdx)
        {
            NDArrayViewElement<f32> const & elementView = inputData.GetDimension(dIdx);
            memcpy(&m_Values.GetElement(0, matrixOffset), elementView.data, elementView.length * sizeof(f32));
            matrixOffset += elementView.length;
        }
    }
}