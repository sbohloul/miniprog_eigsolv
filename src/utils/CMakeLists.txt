add_library(utils)

target_sources(utils
    PRIVATE
    timer.cpp
    array_helper.cpp
    PUBLIC
    timer.hpp
    array_helper.hpp
)

target_include_directories(utils
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)