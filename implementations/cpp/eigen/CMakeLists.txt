cmake_minimum_required(VERSION 3.16)
project(portfolio_eigen CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Require Eigen3 (header-only; install with: sudo apt install libeigen3-dev)
find_package(Eigen3 REQUIRED NO_MODULE)

# Find OpenMP for parallelism over N portfolios
find_package(OpenMP REQUIRED)

# Shared library target
add_library(portfolio_eigen SHARED src/portfolio_compute.cpp)

target_link_libraries(portfolio_eigen
    PRIVATE
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
)

# Aggressive optimisation flags — same as cpp_openmp
target_compile_options(portfolio_eigen PRIVATE
    -O3
    -march=native
    -ffast-math
    -funroll-loops
)

# Unix SO naming convention
set_target_properties(portfolio_eigen PROPERTIES
    PREFIX "lib"
    OUTPUT_NAME "portfolio_eigen"
)
