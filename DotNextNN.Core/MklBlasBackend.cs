using System.Runtime.InteropServices;

namespace DotNextNN.Core
{
    public class MklBlasBackend
    {
        /// <summary>
        /// https://software.intel.com/en-us/mkl-developer-reference-fortran-gemv
        /// 
        /// Computes a matrix-vector product using a general matrix.
        /// y := alpha*A*x + beta*y
        /// </summary>
        public unsafe void gemv(TransposeOptions trans, int m, int n, float alpha, float[] A, int lda, float[] x, int incx, float beta, float[] y, int incy)
        {
            fixed (float* pA = A, pX = x, pY = y)
            {
                cblas_sgemv((int)CblasOrder.CblasColMajor, (int)trans, m, n, alpha, pA, lda, pX, incx, beta, pY, incy);
            }
        }

        /// <summary>
        /// https://software.intel.com/en-us/mkl-developer-reference-fortran-ger
        /// 
        /// Performs a rank-1 update of a general matrix.
        /// A := alpha*x*y'+ A
        /// </summary>
        public unsafe void ger(int m, int n, float alpha, float[] x, int incx, float[] y, int incy, float[] A, int lda)
        {
            fixed (float* px = x, py = y, pA = A)
            {
                cblas_sger((int)CblasOrder.CblasColMajor, m, n, alpha, px, incx, py, incy, pA, lda);
            }
        }

        /// <summary>
        /// https://software.intel.com/en-us/mkl-developer-reference-fortran-axpy
        /// 
        /// Computes a vector-scalar product and adds the result to a vector.
        /// y := a*x + y
        /// </summary>
        public unsafe void axpy(int n, float alpha, float[] x, int incx, float[] y, int incy)
        {
            fixed (float* px = x, py = y)
            {
                cblas_saxpy(n, alpha, px, incx, py, incy);
            }
        }

        /// <summary>
        /// https://software.intel.com/en-us/mkl-developer-reference-fortran-gemm-1
        /// 
        /// Computes a matrix-matrix product with general integer matrices.
        /// C := alpha*(op(A) + A_offset)*(op(B) + B_offset) + beta*C + C_offset 
        /// </summary>
        public unsafe void gemm(TransposeOptions transA, TransposeOptions transB, int m, int n, int k, float alpha, float[] A, int lda, float[] B, int ldb, float beta, float[] C, int ldc)
        {
            fixed (float* pA = A, pB = B, pc = C)
            {
                cblas_sgemm((int)CblasOrder.CblasColMajor, (int)transA, (int)transB, m, n, k, alpha, pA, lda, pB, ldb, beta, pc, ldc);
            }
        }

        #region BLAS interface declaration

        private enum CblasOrder
        {
            CblasRowMajor = 101,
            CblasColMajor = 102
        }

        private const string BLAS_DLL_S = "blas";

        [DllImport(BLAS_DLL_S, CallingConvention = CallingConvention.Cdecl)]
        private static extern unsafe float* cblas_sgemv(int order, int trans, int m, int n, float alpha, float* A,
            int lda, float* x,
            int incx, float beta, float* y, int incy);

        [DllImport(BLAS_DLL_S, CallingConvention = CallingConvention.Cdecl)]
        private static extern unsafe float* cblas_sger(int order, int m, int n, float alpha, float* x,
            int incx, float* y, int incy, float* A, int lda);

        [DllImport(BLAS_DLL_S, CallingConvention = CallingConvention.Cdecl)]
        private static extern unsafe float* cblas_saxpy(int n, float alpha, float* x,
            int incx, float* y, int incy);

        [DllImport(BLAS_DLL_S, CallingConvention = CallingConvention.Cdecl)]
        private static extern unsafe float* cblas_sgemm(int order, int transA, int transB, int m, int n, int k,
            float alpha, float* A, int lda, float* B, int ldb,
            float beta, float* c, int ldc);
        #endregion
    }
}