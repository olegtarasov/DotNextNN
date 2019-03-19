using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DotNextNN.Core.Neural.ErrorFunctions;
using DotNextNN.Core.Optimizers;

namespace DotNextNN.Core.Neural.Layers
{
    public abstract class LayerBase
    {
        protected int BatchSize;

        protected LayerBase()
        {
        }

        public abstract int InputSize { get; }
        public abstract int OutputSize { get; }

        public ErrorFunctionBase ErrorFunction { get; set; }

        public Matrix Input { get; set; } 
        public Matrix Output { get; set; }

        

        /// <summary>
        ///     Forward layer step
        /// </summary>
        /// <param name="input">Input matrix</param>
        /// <param name="inTraining">Store states for back propagation</param>
        /// <returns>Layer output</returns>
        public abstract Matrix Step(Matrix input, bool inTraining = false);

        /// <summary>
        ///     Calculates matched layer error.
        /// </summary>
        /// <param name="y">Layer output</param>
        /// <param name="target">Layer target</param>
        /// <returns></returns>
        public virtual double LayerError(Matrix y, Matrix target)
        {
            if (ErrorFunction == null)
            {
                throw new InvalidOperationException("Layer error function is not specified!");
            }

            return ErrorFunction.GetError(y, target);
        }

        internal void Initialize(int batchSize)
        {
            BatchSize = batchSize;
        }
    }
}