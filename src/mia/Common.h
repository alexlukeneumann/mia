#pragma once

#include <assert.h>
#include <iostream>
#include <initializer_list>
#include <time.h>

namespace mia
{
    typedef signed char             s8;
    typedef signed short int        s16;
    typedef signed long int         s32;
    typedef signed long long int    s64;

    typedef unsigned char           u8;
    typedef unsigned short          u16;
    typedef unsigned long int       u32;
    typedef unsigned long long int  u64;

    typedef float                   f32;
    typedef double                  f64;

    typedef signed char             byte;

    typedef bool                    flag;


#if defined(_DEBUG)

#define ASSERTMSG(cond, msg)                \
    {                                       \
        if (!(cond))                        \
        {                                   \
            std::cout << msg << std::endl;  \
            assert(cond);                   \
        }                                   \
    }

#else

#define ASSERTMSG(cond, msg)

#endif

    // This seed value should be used when requiring a seed. Enables a single point of 
    // change across the library for debugging purposes.
    static u32 const c_SeedValue = static_cast<u32>(time(NULL));
}
