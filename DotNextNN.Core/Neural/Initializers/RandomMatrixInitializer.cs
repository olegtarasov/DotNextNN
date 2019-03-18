using System;

namespace DotNextNN.Core.Neural.Initializers
{
    /// <summary>
    /// Creates random matrices with specified dispersion.
    /// </summary>
    public class RandomMatrixInitializer : IMatrixInitializer
    {
        /// <summary>
        /// Dispersion for random matrices.
        /// </summary>
        public float Dispersion { get; set; } = 5e-2f;

        public Matrix CreateMatrix(int rows, int columns)
        {
            return Matrix.RandomMatrix(rows, columns, Dispersion);
        }
    }
}