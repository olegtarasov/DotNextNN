using System;
using System.Collections.Generic;

namespace DotNextNN.Core.Neural.ErrorFunctions
{
    /// <summary>
    /// Cross-entropy error function.
    /// </summary>
    public class CrossEntropyError : ErrorFunctionBase
    {
        public override double GetError(Matrix output, Matrix target)
        {
            return CrossEntropyErrorImpl(output, target);
        }

        public override Matrix BackpropagateError(Matrix output, Matrix target)
        {
            return BackPropagateCrossEntropyError(output, target);
        }

        private double CrossEntropyErrorImpl(Matrix p, Matrix target)
        {
            if (p.Cols != target.Cols || p.Rows != target.Rows)
                throw new Exception("Matrix dimensions must agree!");

            var pa = (float[])p;
            var ta = (float[])target;

            //E(y0, ... ,yn) = -y0*log(p0)-...-yn*log(pn)
            double err = 0.0d;
            for (int i = 0; i < pa.Length; i++)
            {
                err += ta[i] * Math.Log(pa[i]);
            }

            return -err / p.Cols;
        }

        private Matrix BackPropagateCrossEntropyError(Matrix output, Matrix target)
        {
            var result = new Matrix(output.Rows, output.Cols);
            var oa = (float[])output;
            var ta = (float[])target;
            var ra = (float[])result;

            int cols = output.Cols;
            for (int i = 0; i < oa.Length; i++)
            {
                ra[i] = oa[i] - ta[i];
            }

            for (int i = 0; i < ra.Length; i++)
            {
                ra[i] /= cols;
            }

            return result;
        }
    }
}