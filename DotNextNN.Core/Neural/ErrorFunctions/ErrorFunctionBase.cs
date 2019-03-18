using System;
using System.Collections.Generic;

namespace DotNextNN.Core.Neural.ErrorFunctions
{
    /// <summary>
    /// Base class for the error function. Error function consists of a forward pass function
    /// and a backpropagation function.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class ErrorFunctionBase
    {
        /// <summary>
        /// Gets an error value for single output with respect to specified target.
        /// </summary>
        /// <param name="output">Output matrix.</param>
        /// <param name="target">Target matrix.</param>
        /// <returns>Error value.</returns>
        public abstract double GetError(Matrix output, Matrix target);

        /// <summary>
        /// Propagates the error backwards.
        /// </summary>
        /// <param name="output">Ouput matrix sequence.</param>
        /// <param name="target">Target matrix sequence.</param>
        /// <returns>The sequence of error sensitivities.</returns>
        public abstract Matrix BackpropagateError(Matrix output, Matrix target);

        public abstract ErrorFunctionBase Clone();
    }
}