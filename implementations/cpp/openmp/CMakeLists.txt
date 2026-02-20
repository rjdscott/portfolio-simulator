cmake_minimum_required(VERSION 3.16)
project(portfolio_openmp CXX)

find_package(OpenMP REQUIRED)

# ---------------------------------------------------------------------------
# Helper macro: common properties shared by all targets
# ---------------------------------------------------------------------------
macro(configure_portfolio_target tgt)
    set_target_properties(${tgt} PROPERTIES
        PREFIX "lib"
        OUTPUT_NAME "${tgt}"
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
    target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
endmacro()

# ---------------------------------------------------------------------------
# Target 1: portfolio_openmp (original baseline)
# ---------------------------------------------------------------------------
add_library(portfolio_openmp SHARED src/portfolio_compute.cpp)
target_compile_options(portfolio_openmp PRIVATE
    -O3 -march=native -ffast-math -funroll-loops
)
configure_portfolio_target(portfolio_openmp)

# ---------------------------------------------------------------------------
# Target 2: portfolio_openmp_unroll (8-wide accumulator unroll + day tile)
# Engine: cpp_openmp_unroll
# ---------------------------------------------------------------------------
add_library(portfolio_openmp_unroll SHARED src/portfolio_compute_unroll.cpp)
target_compile_options(portfolio_openmp_unroll PRIVATE
    -O3 -march=native -ffast-math -funroll-loops
)
configure_portfolio_target(portfolio_openmp_unroll)

# ---------------------------------------------------------------------------
# Target 3: portfolio_openmp_tile4 (4-portfolio AVX2 tile, transposed W)
# Engine: cpp_openmp_tile4
# Requires AVX2 + FMA (implied by -march=native on Haswell+; explicit for clarity)
# ---------------------------------------------------------------------------
add_library(portfolio_openmp_tile4 SHARED src/portfolio_compute_tile4.cpp)
target_compile_options(portfolio_openmp_tile4 PRIVATE
    -O3 -march=native -ffast-math -funroll-loops -mavx2 -mfma
)
configure_portfolio_target(portfolio_openmp_tile4)

# ---------------------------------------------------------------------------
# Target 4: portfolio_openmp_clang (identical source, Clang/LLVM compiler)
# Engine: cpp_openmp_clang
#
# Build this target separately with:
#   cmake -S implementations/cpp/openmp \
#         -B implementations/cpp/openmp/build_clang \
#         -DCMAKE_CXX_COMPILER=clang++ \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DBUILD_CLANG_TARGET=ON
#   cmake --build implementations/cpp/openmp/build_clang --parallel
#
# OR (if clang++ is available in the same GCC build directory):
# ---------------------------------------------------------------------------
option(BUILD_CLANG_TARGET "Build the clang-compiled variant" OFF)

if(BUILD_CLANG_TARGET OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_library(portfolio_openmp_clang SHARED src/portfolio_compute.cpp)
    target_compile_options(portfolio_openmp_clang PRIVATE
        -O3 -march=native -ffast-math -funroll-loops
    )
    configure_portfolio_target(portfolio_openmp_clang)
endif()
