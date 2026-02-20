! Portfolio metrics kernel — FORTRAN 2008 + OpenMP + ISO_C_BINDING
!
! Computes cumulative return and annualised Sharpe ratio for N portfolios.
! Uses Welford's online algorithm for numerically stable variance (ddof=1).
!
! Memory layout:
!   Input arrays are C-order (row-major) float64 passed from Python/NumPy.
!   They are declared as explicit-shape Fortran (column-major) arrays with the
!   stock axis (U) as the leading (fastest-varying) dimension.  This makes the
!   inner dot-product loop stride-1 in memory and enables AVX2 vectorisation
!   via gfortran's auto-vectoriser + !$OMP SIMD REDUCTION SIMDLEN(4).
!
!     numpy returns (T, U) C-order  →  Fortran r(U, T):  r(u,t) contiguous on u
!     numpy weights (N, U) C-order  →  Fortran w(U, N):  w(u,i) contiguous on u
!
!   The signature places N, T, U BEFORE the arrays so that the Fortran compiler
!   can see the dimension values when it parses the array declarations.
!
! C ABI (from ctypes):
!   void compute_portfolio_metrics(int64 N, int64 T, int64 U,
!                                  double *r, double *w, double *out);
!
! Build:
!   cmake -S implementations/fortran/openmp \
!         -B implementations/fortran/openmp/build -DCMAKE_BUILD_TYPE=Release
!   cmake --build implementations/fortran/openmp/build --parallel

subroutine compute_portfolio_metrics(N, T, U, r, w, out) BIND(C)
    use ISO_C_BINDING
    implicit none

    integer(C_INT64_T), value :: N, T, U

    ! Explicit-shape arrays: U is leading dimension → stride-1 on u axis.
    ! Interoperable with C's double* via BIND(C).
    real(C_DOUBLE), intent(in)  :: r(U, T)    ! numpy (T, U) C-order → Fortran (U, T)
    real(C_DOUBLE), intent(in)  :: w(U, N)    ! numpy (N, U) C-order → Fortran (U, N)
    real(C_DOUBLE), intent(out) :: out(2, N)  ! output [cum_ret, sharpe] per portfolio

    integer(C_INT64_T) :: i, t_idx, u_idx, cnt
    real(C_DOUBLE)     :: port_r, log_sum, mean_r, m2, delta, delta2

    !$OMP PARALLEL DO DEFAULT(NONE) &
    !$OMP&    SHARED(r, w, out, N, T, U) &
    !$OMP&    PRIVATE(i, t_idx, u_idx, cnt, port_r, log_sum, mean_r, m2, delta, delta2) &
    !$OMP&    SCHEDULE(static)
    do i = 1, N

        log_sum = 0.0_C_DOUBLE
        mean_r  = 0.0_C_DOUBLE
        m2      = 0.0_C_DOUBLE
        cnt     = 0

        do t_idx = 1, T

            ! Dot product: portfolio i vs returns day t_idx.
            ! w(u, i) and r(u, t_idx) both have stride-1 on u
            ! → gfortran emits vmulpd/vaddpd with ymm (AVX2, 4 doubles/cycle).
            port_r = 0.0_C_DOUBLE
            !$OMP SIMD REDUCTION(+:port_r) SIMDLEN(4)
            do u_idx = 1, U
                port_r = port_r + w(u_idx, i) * r(u_idx, t_idx)
            end do

            log_sum = log_sum + port_r

            ! Welford online mean and M2 (ddof=1)
            cnt    = cnt + 1
            delta  = port_r - mean_r
            mean_r = mean_r + delta / real(cnt, C_DOUBLE)
            delta2 = port_r - mean_r
            m2     = m2 + delta * delta2

        end do

        ! Cumulative return: exp(sum of log-returns) - 1
        out(1, i) = exp(log_sum) - 1.0_C_DOUBLE

        ! Annualised Sharpe (ddof=1): mean / sqrt(M2/(T-1)) * sqrt(252)
        if (T > 1 .and. m2 > 0.0_C_DOUBLE) then
            out(2, i) = mean_r / sqrt(m2 / real(T - 1, C_DOUBLE)) * sqrt(252.0_C_DOUBLE)
        else
            out(2, i) = 0.0_C_DOUBLE
        end if

    end do
    !$OMP END PARALLEL DO

end subroutine compute_portfolio_metrics
