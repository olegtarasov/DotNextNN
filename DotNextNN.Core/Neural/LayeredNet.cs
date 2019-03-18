using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DotNextNN.Core.Neural.Layers;
using DotNextNN.Core.Optimizers;


namespace DotNextNN.Core.Neural
{
    public class LayeredNet
    {
        private readonly int _batchSize;
        private readonly int _seqLen;

        private readonly List<LayerBase> _layersList = new List<LayerBase>();

        public LayeredNet(int batchSize, int seqLen, params LayerBase[] layers)
        {
            if (layers.Length == 0) throw new ArgumentException("Value cannot be an empty collection.", nameof(layers));

            _batchSize = batchSize;
            _seqLen = seqLen;

            for (int i = 0; i < layers.Length; i++)
            {
                var layer = layers[i];
                _layersList.Add(layer);
                layer.Initialize(batchSize, seqLen);
                if (i == 0)
                    continue;
               if (layers[i-1].OutputSize != layer.InputSize)
                    throw new ArgumentException($"Dimension of layer #{i} and #{i+1} does not agree ({layers[i-1].OutputSize}!={layer.InputSize})!");
            }
            
        }

        protected LayeredNet(LayeredNet other)
        {
            _batchSize = other._batchSize;
            _seqLen = other._seqLen;
            _layersList = other._layersList.Select(x => x.Clone()).ToList();
        }

        protected LayeredNet(LayeredNet other, int batchSize, int seqLength)
        {
            _batchSize = batchSize;
            _seqLen = seqLength;
            _layersList = other._layersList.Select(x => x.Clone()).ToList();

            foreach (var layer in _layersList)
            {
                layer.Initialize(batchSize, seqLength);
            }
        }

        public OptimizerBase Optimizer { get; set; }

        private LayerBase OutLayer => _layersList[_layersList.Count - 1];
        private LayerBase InLayer => _layersList[0];

        /// <summary>
        /// Clones current network and initializes the new network with specified batch size and sequence length.
        /// </summary>
        /// <param name="batchSize">New batch size.</param>
        /// <param name="seqLength">New sequence length.</param>
        /// <returns>New layered network which is totally decoupled from the source network.</returns>
        public LayeredNet Clone(int batchSize, int seqLength)
        {
            return new LayeredNet(this, batchSize, seqLength);
        }

        public void Optimize()
        {
            foreach (var layer in _layersList)
            {
                layer.Optimize(Optimizer);
            }
        }

        public double Error(Matrix y, Matrix target)
        {
            return OutLayer.LayerError(y, target);
        }

        public Matrix BackPropagate(Matrix target, bool needInputSens = false)
        {  
            Matrix prop = OutLayer.ErrorPropagate(target);
            if (_layersList.Count < 2)
                return prop;
            for (int i = _layersList.Count - 2; i > 0; i--)
            {
                var layer = _layersList[i];
                prop = layer.BackPropagate(prop, true);
            }
            return InLayer.BackPropagate(prop, needInputSens);
        }

        public Matrix Step(Matrix input, bool inTraining = false)
        {
            var prop = input;
            foreach (var layer in _layersList)
                prop = layer.Step(prop, inTraining);
            return prop;
        }

        public void ResetMemory()
        {
            foreach (var layer in _layersList)
                layer.ResetMemory();
        }

        public void ResetOptimizer()
        {
            foreach (var layer in _layersList)
                layer.ResetOptimizer();
        }

        public void Dispose()
        {
        }

        public virtual Matrix Test(Matrix input, Matrix target, out double error)
        {
            var y = Step(input);
            error = Error(y, target);
            
            return y;
        }

        public virtual double Train(Matrix input, Matrix target)
        {
            var y = Step(input, true);
            double error = Error(y, target);
            BackPropagate(target);

            return error;
        }
    }
}