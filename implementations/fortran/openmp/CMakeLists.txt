cmake_minimum_required(VERSION 3.16)
project(portfolio_fortran Fortran)

find_package(OpenMP REQUIRED)

add_library(portfolio_fortran SHARED
    src/portfolio_compute.f90
)

target_compile_options(portfolio_fortran PRIVATE
    -O3
    -march=native
    -ffast-math
    -funroll-loops
    -fPIC
)

target_link_libraries(portfolio_fortran PRIVATE OpenMP::OpenMP_Fortran)

set_target_properties(portfolio_fortran PROPERTIES
    OUTPUT_NAME "portfolio_fortran"
    Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/modules"
)
