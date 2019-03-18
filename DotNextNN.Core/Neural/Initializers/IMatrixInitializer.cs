using System;

namespace DotNextNN.Core.Neural.Initializers
{
    /// <summary>
    /// An generic interface for matrix creation.
    /// </summary>
    public interface IMatrixInitializer
    {
        /// <summary>
        /// Creates a new matrix.
        /// </summary>
        /// <param name="rows">Number of rows.</param>
        /// <param name="columns">Number of columns.</param>
        Matrix CreateMatrix(int rows, int columns);
    }
}