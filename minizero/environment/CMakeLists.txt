file(GLOB_RECURSE SRCS *.cpp)

add_library(environment ${SRCS})

target_include_directories(
    environment PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    base
    kernsearch
    tictactoe
    stochastic
    stochastic/puzzle2048
)

target_link_libraries(
    environment
    config
    utils
    ale::ale-lib
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)
