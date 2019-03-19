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
    }
}