using System;
using System.Collections.Generic;
using System.IO;
using DotNextNN.Core.Neural.ErrorFunctions;
using DotNextNN.Core.Optimizers;

namespace DotNextNN.Core.Neural.Layers
{
    public class LinearLayer : LayerBase
    {
        private NeuroWeight _bias;
        private NeuroWeight _weights;

        public LinearLayer(int xSize, int ySize)
        {
            _weights = new NeuroWeight(Matrix.RandomMatrix(ySize, xSize, 5e-2f));
            _bias = new NeuroWeight(Matrix.RandomMatrix(ySize, 1, 5e-2f));
        }

        public override int InputSize => _weights.Weight.Cols;
        public override int OutputSize => _weights.Weight.Rows;

        public override void Optimize(OptimizerBase optimizer)
        {
            optimizer.Optimize(_weights);
            optimizer.Optimize(_bias);
        }

        public override Matrix Step(Matrix input, bool inTraining = false)
        {
            if (input.Rows != _weights.Weight.Cols)
                throw new Exception($"Wrong input matrix row size provided!\nExpected: {_weights.Weight.Cols}, got: {input.Rows}");
            if (input.Cols != BatchSize)
                throw new Exception($"Wrong input batch size!\nExpected: {BatchSize}, got: {input.Cols}");

            var output = _bias.Weight.TileVector(input.Cols);
            output.Accumulate(_weights.Weight, input);
            if (inTraining)
            {
                Input = input;
                Output = output;
            }
            return output;
        }

        public override Matrix ErrorPropagate(Matrix target)
        {
            return BackPropagate(base.ErrorPropagate(target));
        }

        public override Matrix BackPropagate(Matrix outSens, bool needInputSens = true, bool clearGrad = true)
        {
            if (clearGrad)
            {
                ClearGradients();
            }

            var yIdentity = new Matrix(BatchSize, 1, 1.0f);
            Matrix inputSens = new Matrix(Input.Rows, BatchSize);

            _weights.Gradient.Accumulate(outSens, Input, transposeB: TransposeOptions.Transpose);
            if (BatchSize > 1)
            {
                _bias.Gradient.Accumulate(outSens, yIdentity);
            }
            else
            {
                _bias.Gradient.Accumulate(outSens);
            }

            if (needInputSens)
            {
                inputSens.Accumulate(_weights.Weight, outSens, transposeA: TransposeOptions.Transpose);
            }
            
            return inputSens;
        }

        public override void ClearGradients()
        {
            _weights.ClearGrad();
            _bias.ClearGrad();
        }
    }
}