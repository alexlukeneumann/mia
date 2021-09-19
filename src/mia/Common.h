#pragma once

#include <assert.h>
#include <iostream>

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
}
