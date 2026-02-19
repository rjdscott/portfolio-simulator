cmake_minimum_required(VERSION 3.16)
project(portfolio_openmp CXX)

find_package(OpenMP REQUIRED)

add_library(portfolio_openmp SHARED src/portfolio_compute.cpp)

target_compile_options(portfolio_openmp PRIVATE -O3 -march=native -ffast-math -funroll-loops)

target_link_libraries(portfolio_openmp PRIVATE OpenMP::OpenMP_CXX)

# Produce libportfolio_openmp.so (not liblibportfolio_openmp.so)
set_target_properties(portfolio_openmp PROPERTIES PREFIX "lib" OUTPUT_NAME "portfolio_openmp")

# Hide all symbols by default; only EXPORT-annotated ones are visible
set_target_properties(portfolio_openmp PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
