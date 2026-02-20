// Package main — Portfolio metrics kernel using Go goroutines.
//
// Computes cumulative return and annualised Sharpe ratio for N portfolios.
// Parallelism: a fixed worker pool of runtime.GOMAXPROCS(0) goroutines,
// each processing a contiguous chunk of portfolios.
//
// Note on vectorisation: Go's gc compiler does not auto-vectorise FP
// reductions without explicit math/bits intrinsics.  The inner dot-product
// loop runs scalar.  Performance is dominated by goroutine scheduling overhead
// at small N and by memory bandwidth at large N.
//
// Build as a shared C library:
//   go build -buildmode=c-shared \
//            -o build/libportfolio_go.so \
//            portfolio_compute.go
//
// The -buildmode=c-shared flag also produces a C header (portfolio_compute.h).
package main

/*
#include <stdint.h>
*/
import "C"

import (
	"math"
	"runtime"
	"sync"
	"unsafe"
)

//export ComputePortfolioMetrics
func ComputePortfolioMetrics(
	rPtr *C.double,
	wPtr *C.double,
	outPtr *C.double,
	N C.int64_t,
	T C.int64_t,
	U C.int64_t,
) {
	n := int(N)
	t := int(T)
	u := int(U)

	r   := unsafe.Slice((*float64)(unsafe.Pointer(rPtr)),   t*u)
	w   := unsafe.Slice((*float64)(unsafe.Pointer(wPtr)),   n*u)
	out := unsafe.Slice((*float64)(unsafe.Pointer(outPtr)), n*2)

	sqrt252 := math.Sqrt(252.0)

	nWorkers  := runtime.GOMAXPROCS(0)
	chunkSize := (n + nWorkers - 1) / nWorkers

	var wg sync.WaitGroup

	for start := 0; start < n; start += chunkSize {
		end := start + chunkSize
		if end > n {
			end = n
		}

		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()

			for i := lo; i < hi; i++ {
				wi := w[i*u : (i+1)*u]

				logSum := 0.0
				mean   := 0.0
				m2     := 0.0

				for tIdx := 0; tIdx < t; tIdx++ {
					rRow := r[tIdx*u : (tIdx+1)*u]

					// Dot product — scalar (Go gc does not auto-vectorise FP reductions)
					portR := 0.0
					for k := 0; k < u; k++ {
						portR += wi[k] * rRow[k]
					}

					logSum += portR

					// Welford online mean and M2 (ddof=1)
					count  := float64(tIdx + 1)
					delta  := portR - mean
					mean   += delta / count
					delta2 := portR - mean
					m2     += delta * delta2
				}

				// Cumulative return: exp(sum of log-returns) - 1
				out[i*2] = math.Exp(logSum) - 1.0

				// Annualised Sharpe (ddof=1)
				if t > 1 && m2 > 0.0 {
					variance := m2 / float64(t-1)
					stdR     := math.Sqrt(variance)
					out[i*2+1] = mean / stdR * sqrt252
				} else {
					out[i*2+1] = 0.0
				}
			}
		}(start, end)
	}

	wg.Wait()
}

func main() {}
