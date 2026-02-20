# Julia LoopVectorization Engine

## Setup

Install Julia via juliaup (no sudo required):

```bash
curl -fsSL https://install.julialang.org | sh -s -- --yes
```

Install Julia dependencies (LoopVectorization.jl):

```bash
julia --project=implementations/julia -e 'using Pkg; Pkg.instantiate()'
```

Install the Python bridge:

```bash
uv pip install juliacall
```

juliacall will automatically use the Julia found via juliaup.

## Notes

- The Julia source wraps numpy C-order arrays without copying via juliacall's memory model.
- The `@turbo` macro from LoopVectorization.jl emits SIMD instructions directly (AVX2/FMA).
- `Threads.@threads` uses Julia's thread pool; set `JULIA_NUM_THREADS=auto` for all cores.
- JIT compilation happens on first call (~5-20s); juliacall keeps the session alive.
