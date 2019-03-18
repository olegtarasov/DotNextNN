using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using DotNextNN.Core.Optimizers;

namespace DotNextNN.Core.Neural.Layers
{
    public class SigmoidLayer : LayerBase
    {
        private readonly int _size;

        public SigmoidLayer(int size)
        {
            _size = size;
        }

        public override int InputSize => _size;
        public override int OutputSize => _size;

        public override void Optimize(OptimizerBase optimizer)
        {
        }

        public override Matrix Step(Matrix input, bool inTraining = false)
        {
            var output = input.Clone();

            ApplySigmoid(output);

            if (inTraining)
            {
                Output = output;
            }

            return output;
        }

        public override Matrix ErrorPropagate(Matrix targets)
        {
            return BackPropagate(base.ErrorPropagate(targets));
        }

        public override Matrix BackPropagate(Matrix outSens, bool needInputSens = true, bool clearGrad = true)
        {
            var ones = new Matrix(outSens.Rows, outSens.Cols, 1.0f);
            // osens ^ s(x) ^ (1 - s(x))
            var iSens = (ones - Output) ^ Output ^ outSens;
            return iSens;
        }

        public override void ClearGradients()
        {
        }

        private void ApplySigmoid(Matrix input)
        {
            var m = (float[])input;

            Parallel.For(0, m.Length, i =>
            {
                m[i] = 1.0f / (1 + (float)Math.Exp(-m[i]));
            });
        }
    }
}