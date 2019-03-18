using System;
using System.Threading.Tasks;
using DotNextNN.Core.Neural;

namespace DotNextNN.Core.Optimizers
{
    public class AdamOptimizer : OptimizerBase
    {
        private readonly float _b1, _b2;

        public AdamOptimizer(float learningRate = 1e-3f, float b1 = 0.9f, float b2 = 0.999f) : base(learningRate)
        {
            _b1 = b1;
            _b2 = b2;
        }

        public AdamOptimizer(AdamOptimizer<T> other) : base(other)
        {
            _b1 = other._b1;
            _b2 = other._b2;
        }

        public override void Optimize(NeuroWeight weight)
        {
            weight.Timestep++;

            AdamUpdate(weight);
        }

        private void AdamUpdate(NeuroWeight weight)
        {
            var w = (float[])weight.Weight;
            var c1 = (float[])weight.Cache1;
            var c2 = (float[])weight.Cache2;
            var grad = (float[])weight.Gradient;
            var t = weight.Timestep;

            const float e = 1e-8f;

            Parallel.For(0, w.Length, i =>
            {
                float g = grad[i];

                c1[i] = _b1 * c1[i] + (1 - _b1) * g;
                c2[i] = _b2 * c2[i] + (1 - _b2) * g * g;

                float a = LearningRate * (float)Math.Sqrt(1 - (float)Math.Pow(_b2, t)) / (1 - (float)Math.Pow(_b1, t));
                w[i] = w[i] - a * c1[i] / ((float)Math.Sqrt(c2[i]) + e);
            });
        }
    }
}