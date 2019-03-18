using System;
using System.Collections.Generic;
using System.IO;
using DotNextNN.Core.Neural.ErrorFunctions;
using DotNextNN.Core.Optimizers;
using Retia.RandomGenerator;

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
            var output = SoftMaxNorm(input);
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

        public static List<int> SoftMaxChoice(Matrix p, double T = 1)
        {
            var probs = new List<int>(p.Cols);
            var rnd = SafeRandom.Generator;

            for (int j = 0; j < p.Cols; j++)
            {
                var dChoice = rnd.NextDouble();
                double curPos = 0;
                double nextPos = p[0, j];

                int i;
                for (i = 1; i < p.Rows; i++)
                {
                    if (dChoice > curPos && dChoice <= nextPos)
                        break;
                    curPos = nextPos;
                    nextPos += p[i, j];
                }

                probs.Add(i - 1);
            }
            return probs;
        }

        private Matrix SoftMaxNorm(Matrix y, double T = 1)
        {
            var p = y.Clone();

            var ya = (float[])y;
            var pa = (float[])p;

            var sums = new float[y.Cols];
            for (int i = 0; i < ya.Length; i++)
            {
                pa[i] = (float)Math.Exp(pa[i] / T);
                var c = i / y.Rows;
                sums[c] += pa[i];
            }

            for (int i = 0; i < ya.Length; i++)
            {
                var c = i / y.Rows;
                pa[i] /= sums[c];
            }

            return p;
        }
    }
}