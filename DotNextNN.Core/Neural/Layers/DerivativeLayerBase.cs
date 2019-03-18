using System;
using System.Collections.Generic;
using System.IO;

namespace DotNextNN.Core.Neural.Layers
{
    /// <summary>
    /// Base layer that supports backpropagation of sensitivities with custom element-wise
    /// derivative fuction.
    /// </summary>
    public abstract class DerivativeLayerBase : LayerBase
    {
        protected DerivativeLayerBase()
        {
        }

        protected DerivativeLayerBase(LayerBase other) : base(other)
        {
        }

        protected DerivativeLayerBase(BinaryReader reader) : base(reader)
        {
        }

        /// <summary>
        ///     Get value of layer output derrivative with respect to input (dO/dI of [batch]). Single precision version.
        /// </summary>
        /// <param name="input">Input value matrix</param>
        /// <param name="output">Output value matrix</param>
        /// <param name="batch">Batch index</param>
        /// <param name="i">Input index</param>
        /// <param name="o">Output index</param>
        /// <returns>Derivative value</returns>
        protected abstract float DerivativeS(Matrix input, Matrix output, int batch, int i, int o);

        public override Matrix BackPropagate(Matrix outSens, bool needInputSens = true, bool clearGrad = true)
        {
            return PropagateSensitivity(outSens);
        }

        /// <summary>
        ///     Propagates next layer sensitivity to input, calculating input sensitivity matrix
        /// </summary>
        /// <param name="outSens">Sequence of sensitivity matrices of next layer</param>
        /// <returns></returns>
        protected Matrix PropagateSensitivity(Matrix outSens)
        {
            var iSens = new Matrix(InputSize, BatchSize);
            for (int b = 0; b < BatchSize; b++)
                CalcSens(b, iSens, outSens);
            
            return iSens;
        }

        private void CalcSens(int batch, Matrix iSens, Matrix outSens)
        {
            var isens = iSens;
            var osens = outSens;
            for (int i = 0; i < InputSize; i++)
            {
                for (int o = 0; o < OutputSize; o++)
                {
                    isens[i, batch] += DerivativeS(Input, Output, batch, i, o) * osens[o, batch];
                }
            }

        }
    }
}