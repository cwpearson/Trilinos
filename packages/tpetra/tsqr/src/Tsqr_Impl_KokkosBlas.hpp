// @HEADER
// *****************************************************************************
//          Tpetra: Templated Linear Algebra Services Package
//
// Copyright 2008 NTESS and the Tpetra contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef TSQR_IMPL_KOKKOSBLAS_HPP
#define TSQR_IMPL_KOKKOSBLAS_HPP

#include "KokkosBlas3_gemm.hpp"
#include "KokkosBlas3_trsm.hpp"
#include "KokkosBlas2_gemv.hpp"
#include "Kokkos_Core.hpp"

namespace TSQR {
namespace Impl {

template <class Scalar>
using HostMatLL =
  Kokkos::View<Scalar**, Kokkos::LayoutLeft,
               Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <class Scalar>
using HostVecUnmanaged =
  Kokkos::View<Scalar*, Kokkos::HostSpace,
               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

/// Wrap a contiguous (ld == nrows) const raw pointer as an unmanaged
/// LayoutLeft host View.  Internal use by host_gemm / host_trsm only.
template <class Scalar>
HostMatLL<const Scalar>
make_host_view_unmanaged(const Scalar* ptr, int nrows, int ncols) {
  return HostMatLL<const Scalar>(ptr, nrows, ncols);
}

/// Wrap a contiguous (ld == nrows) mutable raw pointer as an unmanaged
/// LayoutLeft host View.  Internal use by host_gemm / host_trsm only.
template <class Scalar>
HostMatLL<Scalar>
make_host_view_unmanaged(Scalar* ptr, int nrows, int ncols) {
  return HostMatLL<Scalar>(ptr, nrows, ncols);
}

/// Wrap a strided const matrix as an unmanaged LayoutStride host View.
template <class Scalar>
Kokkos::View<const Scalar**, Kokkos::LayoutStride,
             Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
make_host_strided_view(const Scalar* ptr, int nrows, int ncols, int ld) {
  return {ptr, Kokkos::LayoutStride(nrows, 1, ncols, ld)};
}

/// Wrap a strided mutable matrix as an unmanaged LayoutStride host View.
template <class Scalar>
Kokkos::View<Scalar**, Kokkos::LayoutStride,
             Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
make_host_strided_view(Scalar* ptr, int nrows, int ncols, int ld) {
  return {ptr, Kokkos::LayoutStride(nrows, 1, ncols, ld)};
}

/// Wrap a contiguous const vector as an unmanaged host View.
template <class Scalar>
HostVecUnmanaged<const Scalar>
make_host_vec(const Scalar* ptr, int n) {
  return HostVecUnmanaged<const Scalar>(ptr, n);
}

/// Wrap a contiguous mutable vector as an unmanaged host View.
template <class Scalar>
HostVecUnmanaged<Scalar>
make_host_vec(Scalar* ptr, int n) {
  return HostVecUnmanaged<Scalar>(ptr, n);
}

/// Unified host GEMM: C := alpha * op(A) * op(B) + beta * C
///
/// nrows_X / ncols_X are the *stored* extents of X (not the logical shape
/// of op(X)).  For example, if transa='T', A is stored as k-by-m, so pass
/// nrows_A=k, ncols_A=m.
///
/// When all leading dimensions equal their respective row counts (contiguous),
/// unmanaged LayoutLeft Views are used.  Otherwise, unmanaged LayoutStride
/// Views wrap the raw pointers directly, avoiding any extra copies.
template <class Scalar>
void host_gemm(const char transa, const char transb,
               const Scalar& alpha,
               const Scalar* A, int nrows_A, int ncols_A, int lda,
               const Scalar* B, int nrows_B, int ncols_B, int ldb,
               const Scalar& beta,
               Scalar* C, int nrows_C, int ncols_C, int ldc)
{
  const char ta[2] = {transa, '\0'};
  const char tb[2] = {transb, '\0'};
  if (lda == nrows_A && ldb == nrows_B && ldc == nrows_C) {
    KokkosBlas::gemm(ta, tb, alpha,
                     make_host_view_unmanaged(A, nrows_A, ncols_A),
                     make_host_view_unmanaged(B, nrows_B, ncols_B),
                     beta,
                     make_host_view_unmanaged(C, nrows_C, ncols_C));
  } else {
    KokkosBlas::gemm(ta, tb, alpha,
                     make_host_strided_view(A, nrows_A, ncols_A, lda),
                     make_host_strided_view(B, nrows_B, ncols_B, ldb),
                     beta,
                     make_host_strided_view(C, nrows_C, ncols_C, ldc));
  }
}

/// Unified host TRSM: B := alpha * op(A)^{-1} * B  (side == 'L')
///                or  B := alpha * B * op(A)^{-1}  (side == 'R')
///
/// adim is the dimension of the square triangular matrix A.
/// nrows_B / ncols_B are the stored extents of B.
template <class Scalar>
void host_trsm(const char side, const char uplo,
               const char transa, const char diag,
               const Scalar& alpha,
               const Scalar* A, int adim, int lda,
               Scalar* B, int nrows_B, int ncols_B, int ldb)
{
  const char si[2] = {side,   '\0'};
  const char ul[2] = {uplo,   '\0'};
  const char ta[2] = {transa, '\0'};
  const char di[2] = {diag,   '\0'};
  if (lda == adim && ldb == nrows_B) {
    KokkosBlas::trsm(si, ul, ta, di, alpha,
                     make_host_view_unmanaged(A, adim, adim),
                     make_host_view_unmanaged(B, nrows_B, ncols_B));
  } else {
    KokkosBlas::trsm(si, ul, ta, di, alpha,
                     make_host_strided_view(A, adim, adim, lda),
                     make_host_strided_view(B, nrows_B, ncols_B, ldb));
  }
}

}  // namespace Impl
}  // namespace TSQR

#endif  // TSQR_IMPL_KOKKOSBLAS_HPP
