#include <Models/Sequential.h>
#include <Layers/Flatten.h>
#include <Layers/Dense.h>

using namespace mia;

int main()
{
    // The following code demonstrates using mia to model an XOR gate.
    activators::ActivatorType const type = activators::ActivatorType::Sigmoid;

    Sequential model({
        new Flatten({ 2 }, type),
        new Dense(2, type),
        new Dense(1, type)
    });

    // Compile the model
    model.Compile(c_SeedValue);

    // Create the input data & the models expected output
    f32 inputData[] = {
        0.0f, 0.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f
    };

    f32 expectedOutput[] = {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };

    // Train the model
    u32 const numIterations = 1000;
    for (u32 iIdx = 0; iIdx < numIterations; ++iIdx)
    {
        u32 const numInputData = LENGTHOF(inputData) / 2;
        for (u32 i = 0; i < numInputData; ++i)
        {
            model.Train(
                {{ static_cast<DimensionLength>(2), &inputData[i * 2] }}, 
                { expectedOutput[i] }
            );
        }
    }

    return 0;
}
