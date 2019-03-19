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