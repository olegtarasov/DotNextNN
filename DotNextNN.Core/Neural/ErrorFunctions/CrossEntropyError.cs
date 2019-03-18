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

        public override ErrorFunctionBase Clone()
        {
            return new CrossEntropyError();
        }

        private double CrossEntropyErrorImpl(Matrix p, Matrix target)
        {
            if (p.Cols != target.Cols || p.Rows != target.Rows)
                throw new Exception("Matrix dimensions must agree!");

            //var pa = p.AsColumnMajorArray();
            //var ta = target.AsColumnMajorArray();
            var pa = (float[])p;
            var ta = (float[])target;

            //E(y0, ... ,yn) = -y0*log(p0)-...-yn*log(pn)
            double err = 0.0d;
            int notNan = 0;
            int cols = p.Cols;
            for (int i = 0; i < pa.Length; i++)
            {
                if (i > 0 && i % p.Rows== 0)
                {
                    if (notNan == 0)
                        cols--;

                    notNan = 0;
                }

                if (float.IsNaN(ta[i]))
                    continue;

                notNan++;

                err += ta[i] * Math.Log(pa[i]);
            }

            if (cols == 0)
            {
                throw new InvalidOperationException("All of your targets are NaN! This is pointless.");
            }

            return -err / p.Cols;
        }

        private Matrix BackPropagateCrossEntropyError(Matrix output, Matrix target)
        {
            var result = new Matrix(output.Rows, output.Cols);
            var oa = (float[])output;
            var ta = (float[])target;
            var ra = (float[])result;

            int rows = output.Rows;
            int cols = output.Cols;
            int notNan = 0;
            for (int i = 0; i < oa.Length; i++)
            {
                if (i > 0 && i % rows == 0)
                {
                    if (notNan == 0)
                        cols--;

                    notNan = 0;
                }

                if (float.IsNaN(ta[i]))
                {
                    continue;
                }

                notNan++;

                ra[i] = oa[i] - ta[i];
            }

            if (cols == 0)
            {
                throw new InvalidOperationException("All of your targets are NaN! This is pointless.");
            }

            for (int i = 0; i < ra.Length; i++)
            {
                ra[i] /= cols;
            }

            return result;
        }
    }
}