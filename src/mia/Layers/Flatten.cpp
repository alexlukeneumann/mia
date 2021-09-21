#include "Flatten.h"

namespace mia
{
    Flatten::~Flatten()
    {
        if (nullptr != m_InputDimensionLengths)
        {
            delete[] m_InputDimensionLengths;
        }
    }

    Flatten::Flatten(std::initializer_list<DimensionLength> const & inputDimensionLengths)
        : m_InputNumDimensions(static_cast<u32>(inputDimensionLengths.size()))
        , m_InputDimensionLengths(nullptr)
    {
        ASSERTMSG(m_InputNumDimensions != 0, "Cannot create a Flatten layer with expected data containing 0 dimensions.");
        m_InputDimensionLengths = new u32[m_InputNumDimensions];
        memcpy(m_InputDimensionLengths, inputDimensionLengths.begin(), m_InputNumDimensions * sizeof(u32));
    }

    void Flatten::Compile(u32 seedValue, Layer const * prevLayer)
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

        // Specifically mark biases & weights to be empty as the Flatten layer doens't use them.
        m_Biases = Matrix();
        m_Weights = Matrix();
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
        u32 columnOffset = 0;
        for (u32 dIdx = 0; dIdx < inputData.GetNumDimensions(); ++dIdx)
        {
            NDArrayViewElement<f32> const & elementView = inputData.GetDimension(dIdx);
            m_Values.Copy(0, columnOffset, elementView.data, elementView.length * sizeof(f32));
            columnOffset += elementView.length;
        }
    }
}
