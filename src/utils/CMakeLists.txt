add_library(utils)

target_sources(utils
    PRIVATE
    timer.cpp
    array_helper.cpp
)

target_include_directories(utils
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)