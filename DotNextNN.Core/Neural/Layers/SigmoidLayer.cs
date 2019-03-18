using System;
using System.Collections.Generic;
using System.IO;
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

        public SigmoidLayer(BinaryReader reader) : base(reader)
        {
            _size = reader.ReadInt32();
        }

        private SigmoidLayer(SigmoidLayer other) : base(other)
        {
            _size = other._size;
        }

        public override int InputSize => _size;
        public override int OutputSize => _size;
        public override int TotalParamCount => 0;

        public override LayerBase Clone()
        {
            return new SigmoidLayer(this);
        }

        public override void Optimize(OptimizerBase optimizer)
        {
        }

        public override Matrix Step(Matrix input, bool inTraining = false)
        {
            var output = input.Clone();

            MathProvider.ApplySigmoid(output);

            if (inTraining)
            {
                Outputs.Add(output);
            }

            return output;
        }

        public override List<Matrix> ErrorPropagate(List<Matrix> targets)
        {
            return BackPropagate(base.ErrorPropagate(targets));
        }

        public override List<Matrix> BackPropagate(List<Matrix> outSens, bool needInputSens = true, bool clearGrad = true)
        {
            if (Outputs.Count == 0)
                throw new Exception("Empty inputs history, nothing to propagate!");
            if (outSens.Count != Outputs.Count)
                throw new Exception("Not enough sensitivies in list!");

            var ones = new Matrix(outSens[0].Rows, outSens[0].Cols, 1.0f);
            var iSens = new List<Matrix>();
            for (int t = 0; t < outSens.Count; t++)
            {
                var output = Outputs[t];
                var osens = outSens[t];

                // osens ^ s(x) ^ (1 - s(x))
                iSens.Add((ones - output) ^ output ^ osens);
            }
            return iSens;
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
        }

        public override void ClampGrads(float limit)
        {
        }

        public override void ToVectorState(T[] destination, ref int idx, bool grad = false)
        {
        }

        public override void FromVectorState(T[] vector, ref int idx)
        {
        }

        public override void ClearGradients()
        {
        }
    }
}