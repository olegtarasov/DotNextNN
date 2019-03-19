using System;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using Retia.RandomGenerator;

namespace DotNextNN.Core
{
    public enum TransposeOptions
    {
        NoTranspose = 111,
        Transpose = 112,
    }

    public class Matrix
    {
        private static readonly MklBlasBackend _backend = new MklBlasBackend();

        public readonly int Rows;
        public readonly int Cols;
        public readonly int Length;

        private readonly float[] _storage;

        public Matrix(int rows, int cols) : this(rows, cols, new float[rows * cols])
        {
        }

        public Matrix(int rows, int cols, float value) : this(rows, cols, Enumerable.Repeat(value, rows * cols).ToArray())
        {
        }

        private Matrix(int rows, int cols, float[] storage)
        {
            Rows = rows;
            Cols = cols;
            Length = Rows * Cols;
            _storage = storage;
        }

        
        public float this[int row, int col]
        {
            get
            {
                CheckBounds(row, col);
                return _storage[col * Rows + row];
            }
            set
            {
                CheckBounds(row, col);
                _storage[col * Rows + row] = value;
            }
        }

        public static Matrix FromColumnArrays(float[][] source)
        {
            var storage = new float[source[0].Length * source.Length];
            int idx = -1;

            for (int col = 0; col < source.Length; col++)
            {
                for (int row = 0; row < source[0].Length; row++)
                {
                    storage[++idx] = source[col][row];
                }
            }

            return new Matrix(source[0].Length, source.Length, storage);
        }

        public static Matrix RandomMatrix(int rows, int cols, float min, float max)
        {
            var random = SafeRandom.Generator;
            var matrix = new Matrix(rows, cols);
            for (int i = 0; i < matrix.Length; i++)
                matrix._storage[i] = (float)random.NextDouble(min, max);
            return matrix;
        }

        public static Matrix RandomMatrix(int rows, int cols, float dispersion)
        {
            return RandomMatrix(rows, cols, -dispersion, dispersion);
        }

        public static implicit operator float[] (Matrix m)
        {
            return m._storage;
        }

        public static Matrix operator -(Matrix m)
        {
            return Multiply(-1, m);
        }

        public static Matrix operator +(Matrix m1, Matrix m2)
        {
            return Add(m1, m2);
        }

        public static Matrix operator -(Matrix m1, Matrix m2)
        {
            return Add(m1, -m2);
        }

        public static Matrix operator *(float n, Matrix m)
        {
            return Multiply(n, m);
        }

        public static Matrix operator *(Matrix m1, Matrix m2)
        {
            var result = new Matrix(m1.Rows, m2.Cols);

            result.Accumulate(m1, m2);
            return result;
        }

        public static Matrix operator ^(Matrix m1, Matrix m2)
        {
            return HadamardMul(m1, m2);
        }

        public Matrix Clone()
        {
            var matrix = new Matrix(Rows, Cols);
            Array.Copy(_storage, matrix._storage, _storage.Length);

            return matrix;
        }

        public void Clear()
        {
            Array.Clear(_storage, 0, _storage.Length);
        }

        public Matrix TileVector(int cols)
        {
            if (Cols > 1)
            {
                throw new InvalidOperationException("Source matrix is not a column vector!");
            }

            var matrix = new Matrix(Rows, cols);

            for (int col = 0; col < cols; col++)
            {
                int offset = col * Rows;
                Array.Copy(_storage, 0, matrix._storage, offset, Rows);
            }

            return matrix;
        }

        /// <summary>
        ///     this = beta*this + alpha*AB;
        /// </summary>
        public void Accumulate(Matrix A, Matrix B, float beta = 0.0f, float alpha = 1.0f, TransposeOptions transposeA = TransposeOptions.NoTranspose, TransposeOptions transposeB = TransposeOptions.NoTranspose)
        {
            if ((B.Cols > 1 && transposeB == TransposeOptions.NoTranspose) || (B.Rows > 1 && transposeB == TransposeOptions.Transpose))
            {
                DotMatrix(A, B, this, beta, alpha, transposeA, transposeB);
            }
            else
            {
                if (A.Cols > 1)
                    DotVec(A, B, this, beta, alpha, transposeA);
                else
                    UpdMatFromVec(A, B, this, alpha);
            }
        }

        /// <summary>
        ///     this = alpha*A+this
        /// </summary>
        public void Accumulate(Matrix A, float alpha = 1.0f)
        {
            if (A.Cols == 1)
            {
                SumVec(A, this, alpha);
            }
            else
            {
                var result = alpha == 1.0d ? A : alpha * A;
                Array.Copy((result + this)._storage, _storage, _storage.Length);
            }
        }

        /// <summary>
        ///     A = alpha*xyT + A
        /// </summary>
        private static unsafe void UpdMatFromVec(Matrix x, Matrix y, Matrix A, float alpha = 1.0f)
        {
            if (x.Rows != y.Cols)
            {
                throw new InvalidOperationException("Vector dimensions do not agree.");
            }

            _backend.ger(x.Rows, y.Rows, alpha, x, 1, y, 1, A, A.Rows);
        }

        /// <summary>
        ///     y = beta*y + alpha*Ax;
        /// </summary>
        private static unsafe void DotVec(Matrix A, Matrix x, Matrix y, float beta, float alpha, TransposeOptions transposeA)
        {
            int aCols = transposeA == TransposeOptions.NoTranspose ? A.Cols : A.Rows;

            if (aCols != x.Rows)
            {
                throw new InvalidOperationException("Matrix and vector dimensions do not agree.");
            }

            _backend.gemv(transposeA, A.Rows, A.Cols, alpha, A._storage, A.Rows, x._storage, 1, beta, y._storage, 1);
        }

        /// <summary>
        ///     C = alpha*AB + beta*C
        /// </summary>
        private static unsafe void DotMatrix(Matrix A, Matrix B, Matrix C, float beta = 0.0f, float alpha = 1.0f, TransposeOptions transposeA = TransposeOptions.NoTranspose, TransposeOptions transponseB = TransposeOptions.NoTranspose)
        {
            int m = C.Rows;
            int n = C.Cols;
            int k = (transposeA == TransposeOptions.NoTranspose) ? A.Cols : A.Rows;
            int bRows = (transponseB == TransposeOptions.NoTranspose) ? B.Rows : B.Cols;

            if (k != bRows)
            {
                throw new InvalidOperationException("Matrix dimensions don't agree.");
            }

            _backend.gemm(transposeA, transponseB, m, n, k, alpha, A, A.Rows, B, B.Rows, beta, C, C.Rows);
        }

        /// <summary>
        ///     y=alpha*x + y;
        /// </summary>
        private static unsafe void SumVec(Matrix x, Matrix y, float alpha)
        {
            if (y.Cols > 1 || x.Cols > 1)
                throw new Exception("Vector BLAS function is called with matrix argument!");

            if (y.Rows != x.Rows)
                throw new Exception("Vector dimensions must agree!");

            _backend.axpy(x.Rows, alpha, x, 1, y, 1);
        }

        private static Matrix Multiply(float n, Matrix m)
        {
            var r = new Matrix(m.Rows, m.Cols);
            for (int i = 0; i < m.Length; i++)
                r._storage[i] = m._storage[i] * n;
            return r;
        }

        private static Matrix Add(Matrix m1, Matrix m2)
        {
            if (m1.Rows != m2.Rows || m1.Cols != m2.Cols) throw new InvalidOperationException("Matrices must have the same dimensions!");

            var r = new Matrix(m1.Rows, m1.Cols);
            for (int i = 0; i < m1.Length; i++)
                r._storage[i] = m1._storage[i] + m2._storage[i];
            return r;
        }

        private static Matrix HadamardMul(Matrix m1, Matrix m2)
        {
            if (m1.Rows != m2.Rows || m1.Cols != m2.Cols) throw new InvalidOperationException("Matrices must have the same dimensions!");
            var r = new Matrix(m1.Rows, m1.Cols);
            for (int i = 0; i < m1.Length; i++)
                r._storage[i] = m1._storage[i] * m2._storage[i];
            return r;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void CheckBounds(int row, int col)
        {
            if (row < 0 || row >= Rows)
            {
                throw new InvalidOperationException("Matrix row is out of range.");
            }

            if (col < 0 || col >= Cols)
            {
                throw new InvalidOperationException("Matrix column is out of range.");
            }
        }
    }
}