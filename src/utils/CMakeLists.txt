add_library(utils)

target_sources(utils
    PRIVATE
    timer.cpp
    PUBLIC
    timer.hpp
)

target_include_directories(utils
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)