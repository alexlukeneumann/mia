#include <Models/Sequential.h>
#include <Layers/Flatten.h>
#include <Layers/Dense.h>

int main()
{
    // The following code demonstrates using mia to model an XOR gate.
    mia::Sequential model({
        new mia::Flatten({ 2 }),
        new mia::Dense(2),
        new mia::Dense(1)
    });

    // Compile the model
    model.Compile(mia::c_SeedValue);

    // Create the input data & the models expected output
    mia::f32 inputData[] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };

    mia::f32 expectedOutput[] = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };

    // Train the model
    mia::u32 const numIterations = 1000;
    for (mia::u32 iIdx = 0; iIdx < numIterations; ++iIdx)
    {
        mia::u32 const numInputData = LENGTHOF(inputData) / 2;
        for (mia::u32 i = 0; i < numInputData; ++i)
        {
            model.Train(
                {{ static_cast<mia::DimensionLength>(2), &inputData[i * 2] }}, 
                { expectedOutput[i] }
            );
        }
    }

    return 0;
}
