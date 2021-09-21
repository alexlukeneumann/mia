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

set(MIA_CORE_FILES
  src/mia/Core/NDArrayView.h
)

set(MIA_MATH_FILES
  src/mia/Maths/Matrix.h
  src/mia/Maths/Matrix.cpp
)

set(MIA_LAYERS_FILES
  src/mia/Layers/Layer.h
  src/mia/Layers/InputLayer.h
  src/mia/Layers/Flatten.h
  src/mia/Layers/Flatten.cpp
  src/mia/Layers/Dense.h
  src/mia/Layers/Dense.cpp
)

SOURCE_GROUP(src FILES ${MIA_SRC_FILES})
SOURCE_GROUP(src/Maths FILES ${MIA_MATH_FILES})
SOURCE_GROUP(src/Core FILES ${MIA_CORE_FILES})
SOURCE_GROUP(src/Layers FILES ${MIA_LAYERS_FILES})

add_library(mia STATIC
  ${MIA_SRC_FILES}
  ${MIA_MATH_FILES}
  ${MIA_CORE_FILES}
  ${MIA_LAYERS_FILES}
)

target_include_directories(mia PUBLIC src/mia)

##################################################
## Mia Tests
##################################################

project(mia_tests)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(MIA_CORE_TEST_FILES
  src/mia_tests/Core/NDArrayView.tests.cpp
)

set(MIA_MATHS_TEST_FILES
  src/mia_tests/Maths/Matrix.tests.cpp
)

set(MIA_LAYERS_TEST_FILES
  src/mia_tests/Layers/Layer.tests.cpp
  src/mia_tests/Layers/Flatten.tests.cpp
  src/mia_tests/Layers/Dense.tests.cpp
)

set(MIA_LAYERS_HELPERS_TEST_FILES
  src/mia_tests/Layers/Helpers/LayerManipulator.h
  src/mia_tests/Layers/Helpers/LayerManipulator.cpp
  src/mia_tests/Layers/Helpers/TestInputLayer.h
  src/mia_tests/Layers/Helpers/TestInputLayer.cpp
  src/mia_tests/Layers/Helpers/TestInputLayer.tests.cpp
)

SOURCE_GROUP(src/Core FILES ${MIA_CORE_TEST_FILES})
SOURCE_GROUP(src/Maths FILES ${MIA_MATHS_TEST_FILES})
SOURCE_GROUP(src/Layers FILES ${MIA_LAYERS_TEST_FILES})
SOURCE_GROUP(src/Layers/Helpers FILES ${MIA_LAYERS_HELPERS_TEST_FILES})

add_library(mia_tests SHARED
  ${MIA_CORE_TEST_FILES}
  ${MIA_MATHS_TEST_FILES}
  ${MIA_LAYERS_TEST_FILES}
  ${MIA_LAYERS_HELPERS_TEST_FILES}
)

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
