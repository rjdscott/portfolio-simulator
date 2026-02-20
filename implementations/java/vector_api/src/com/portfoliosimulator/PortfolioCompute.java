package com.portfoliosimulator;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.stream.IntStream;

/**
 * Portfolio metrics kernel using the Java Vector API (jdk.incubator.vector).
 *
 * <p>Computes cumulative return and annualised Sharpe ratio for N portfolios.
 * Parallelism: {@code IntStream.range(0, N).parallel()} dispatches to the
 * {@code ForkJoinPool.commonPool()}, which sizes to available processors.
 *
 * <p>SIMD: The inner dot-product over U=100 stocks uses explicit 256-bit AVX2
 * lanes ({@code SPECIES_256} = 4 doubles per lane).  The {@code reduceLanes(ADD)}
 * operation produces the horizontal sum.  Unlike Java auto-vectorisation, the
 * Vector API gives the programmer direct control over lane width and operation
 * type — no {@code -ffast-math} analogue required.
 *
 * <p>Build:
 * <pre>
 *   javac --add-modules jdk.incubator.vector \
 *         -d dist/classes \
 *         src/com/portfoliosimulator/PortfolioCompute.java
 *   jar cf dist/portfolio_vector_api.jar -C dist/classes .
 * </pre>
 *
 * <p>Called from Python via JPype:
 * <pre>
 *   import jpype, jpype.imports
 *   jpype.startJVM(classpath=["dist/portfolio_vector_api.jar"],
 *                  convertStrings=False,
 *                  "--add-modules", "jdk.incubator.vector")
 *   from com.portfoliosimulator import PortfolioCompute
 *   PortfolioCompute.computeMetrics(r, w, out, N, T, U)
 * </pre>
 */
public class PortfolioCompute {

    // 256-bit lane width = 4 doubles per operation (AVX2)
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_256;
    private static final double SQRT_252 = Math.sqrt(252.0);

    /**
     * Compute cumulative return and annualised Sharpe for N portfolios.
     *
     * @param r   flat (T * U) double[] of log-returns, C-order (row = day)
     * @param w   flat (N * U) double[] of portfolio weights, C-order (row = portfolio)
     * @param out flat (N * 2) double[] output buffer: [cum_return, sharpe] per portfolio
     * @param N   number of portfolios
     * @param T   number of trading days
     * @param U   universe size (number of stocks)
     */
    public static void computeMetrics(
        double[] r, double[] w, double[] out,
        int N, int T, int U
    ) {
        IntStream.range(0, N).parallel().forEach(i -> {
            final int wBase = i * U;

            double logSum = 0.0;
            double mean   = 0.0;
            double m2     = 0.0;

            for (int t = 0; t < T; t++) {
                final int rBase = t * U;

                // Vectorised dot product using 256-bit AVX2 lanes (4 doubles each)
                double portR = 0.0;
                int u = 0;
                int limit = SPECIES.loopBound(U);

                for (; u < limit; u += SPECIES.length()) {
                    DoubleVector vw = DoubleVector.fromArray(SPECIES, w, wBase + u);
                    DoubleVector vr = DoubleVector.fromArray(SPECIES, r, rBase + u);
                    portR += vw.mul(vr).reduceLanes(VectorOperators.ADD);
                }
                // Scalar tail (handles U not divisible by lane width)
                for (; u < U; u++) {
                    portR += w[wBase + u] * r[rBase + u];
                }

                logSum += portR;

                // Welford online mean and M2 (ddof=1)
                double count  = t + 1.0;
                double delta  = portR - mean;
                mean         += delta / count;
                double delta2 = portR - mean;
                m2           += delta * delta2;
            }

            // Cumulative return: expm1 for numerical stability at small sums
            out[i * 2]     = Math.expm1(logSum);

            // Annualised Sharpe (ddof=1): mean / std * sqrt(252)
            if (T > 1 && m2 > 0.0) {
                double variance = m2 / (T - 1.0);
                double stdR     = Math.sqrt(variance);
                out[i * 2 + 1] = mean / stdR * SQRT_252;
            } else {
                out[i * 2 + 1] = 0.0;
            }
        });
    }
}
