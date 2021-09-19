cmake_minimum_required(VERSION 3.10)

##################################################
## Mia
##################################################

project(mia)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(MIA_SRC_FILES
  src/mia/Common.h
)

set(MIA_MATH_FILES
  src/mia/Maths/Matrix.h
  src/mia/Maths/Matrix.cpp
)

SOURCE_GROUP(src FILES ${MIA_SRC_FILES})
SOURCE_GROUP(src/Maths FILES ${MIA_MATH_FILES})

add_library(mia STATIC
  ${MIA_SRC_FILES}
  ${MIA_MATH_FILES}
)

target_include_directories(mia PUBLIC src/mia)

##################################################
## Mia Tests
##################################################

project(mia_tests)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(MIA_TEST_FILES
  src/mia_tests/Maths/Matrix.tests.cpp
)

SOURCE_GROUP(src/Maths FILES ${MIA_TEST_FILES})

add_library(mia_tests SHARED ${MIA_TEST_FILES})

target_link_libraries(mia_tests mia)
target_include_directories(mia_tests PUBLIC src)

##################################################
## Demo
##################################################

project(demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set (DEMO_SRC_FILES
  src/demo/main.cpp
)

SOURCE_GROUP(src FILES ${DEMO_SRC_FILES})

add_executable(demo ${DEMO_SRC_FILES})

target_link_libraries(demo mia)
target_include_directories(demo PUBLIC src)
