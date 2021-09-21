#include "TestInputLayer.h"

namespace mia
{
    namespace Tests
    {
        TestInputLayer::TestInputLayer(u32 numNeurons)
            : m_NumNeurons(numNeurons)
        {
        }

        void TestInputLayer::Compile(u32 seedValue, Layer const * prevLayer)
        {
            m_Values = Matrix(1, m_NumNeurons);
        }

        void TestInputLayer::SetInputData(NDArrayView<f32> const & inputData)
        {
            ASSERTMSG(1 == inputData.GetNumDimensions(), "TestInputLayer expects a 1D input array.");
            ASSERTMSG(m_NumNeurons == inputData.GetDimension(0).length, "TestInputLayer's number of neurons differs to the supplied data.");

            f32 const * src = inputData.GetDimension(0).data;
            m_Values.Copy(0, 0, src, m_NumNeurons);
        }
    }
}
