#include <Layers/Flatten.h>
#include <Layers/Dense.h>

int main()
{
    // XOR layer setup
    mia::Layer* layers[] = {
        new mia::Flatten({ 2 }),
        new mia::Dense(2),
        new mia::Dense(1)
    };

    // Cleanup
    for (mia::u64 lIdx = 0; lIdx < LENGTHOF(layers); ++lIdx)
    {
        delete layers[lIdx];
    }

    return 0;
}
