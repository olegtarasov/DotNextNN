using System;
using System.IO;
using DotNextNN.Core.Neural.ErrorFunctions;
using DotNextNN.Core.Optimizers;

namespace DotNextNN.Core.Neural.Layers
{
    public class SoftMaxLayer : DerivativeLayerBase
    {
        private readonly int _size;

        private SoftMaxLayer(SoftMaxLayer other) : base(other)
        {
            _size = other._size;
        }

        public SoftMaxLayer(int size)
        {
            _size = size;

            ErrorFunction = new CrossEntropyError();
        }

        public SoftMaxLayer(BinaryReader reader) : base(reader)
        {
            _size = reader.ReadInt32();
        }

        public override int InputSize => _size;
        public override int OutputSize => _size;
        public override int TotalParamCount => 0;

        public override LayerBase Clone()
        {
            return new SoftMaxLayer(this);
        }

        public override void Optimize(OptimizerBase optimizer)
        {
        }

        public override Matrix Step(Matrix input, bool inTraining = false)
        {
            var output = MathProvider.SoftMaxNorm(input);
            if (inTraining)
            {
                Inputs.Add(input);
                Outputs.Add(output);
            }
            return output;
        }

        protected override float DerivativeS(Matrix input, Matrix output, int batch, int i, int o)
        {
            return i == o ? output[i, batch] * (1 - output[o, batch]) : -output[i, batch] * output[o, batch];
        }

        public override void ResetMemory()
        {
        }

        public override void ResetOptimizer()
        {
        }

        public override void InitSequence()
        {
            Outputs.Clear();
            Inputs.Clear();
        }

        public override void ClampGrads(float limit)
        {
        }

        public override void ClearGradients()
        {
        }
    }
}