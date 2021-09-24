#include <Models/Sequential.h>
#include <Layers/Flatten.h>
#include <Layers/Dense.h>

#include <iostream>

using namespace mia;

int main()
{
    // The following code demonstrates using mia to model an XOR gate.
    activators::ActivatorType const type = activators::ActivatorType::ReLU;

    models::Sequential model({
        new layers::Flatten({ 2 }, type),
        new layers::Dense(2, type),
        new layers::Dense(1, type)
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
    for (u32 i = 0; i < 20; ++i)
    {
        f32 mse = 0.0f;

        u32 const numIterations = 10000;
        for (u32 iIdx = 0; iIdx < numIterations; ++iIdx)
        {
            u32 const numInputData = LENGTHOF(inputData) / 2;
            for (u32 i = 0; i < numInputData; ++i)
            {
                mse += model.Train(
                    {{ static_cast<DimensionLength>(2), &inputData[i * 2] }}, 
                    { expectedOutput[i] }
                );
            }
        }

        std::cout << "MSE: " << mse / (numIterations * 4) << "\n";
    }

    // Execute model
    u32 const numInputData = LENGTHOF(inputData) / 2;
    for (u32 i = 0; i < numInputData; ++i)
    {
        Matrix const result = model.Execute(
            {{ static_cast<DimensionLength>(2), &inputData[i * 2] }}
        );

        char buffer[128];
        sprintf_s(buffer, "Input: (%.2f, %.2f) | Output: (%.2f)", inputData[(i * 2) + 0], inputData[(i * 2) + 1], result.GetElement(0, 0));

        std::cout << buffer << std::endl;
    }

    return 0;
}
