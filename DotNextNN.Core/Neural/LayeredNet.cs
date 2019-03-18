using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DotNextNN.Core.Neural.Layers;
using DotNextNN.Core.Optimizers;


namespace DotNextNN.Core.Neural
{
    public class LayeredNet : NeuralNet
    {
        private const byte LayerMagic = 0xBA;
        //public static uint cnt = 0;
        //public readonly uint id; 
        private static readonly byte[] _magic = {0xDE, 0xAD, 0xCA, 0xFE};

        protected readonly int BatchSize, SeqLen;
        protected readonly List<LayerBase> LayersList = new List<LayerBase>();

        private IntPtr _gpuNetworkPtr = IntPtr.Zero;
        private OptimizerBase _optimizer;

        public LayeredNet(int batchSize, int seqLen, params LayerBase[] layers)
        {
            if (layers.Length == 0) throw new ArgumentException("Value cannot be an empty collection.", nameof(layers));

            BatchSize = batchSize;
            SeqLen = seqLen;

            for (int i = 0; i < layers.Length; i++)
            {
                var layer = layers[i];
                LayersList.Add(layer);
                layer.Initialize(batchSize, seqLen);
                if (i == 0)
                    continue;
               if (layers[i-1].OutputSize != layer.InputSize)
                    throw new ArgumentException($"Dimension of layer #{i} and #{i+1} does not agree ({layers[i-1].OutputSize}!={layer.InputSize})!");
            }
            
        }

        protected LayeredNet(LayeredNet other)
        {
            BatchSize = other.BatchSize;
            SeqLen = other.SeqLen;
            LayersList = other.LayersList.Select(x => x.Clone()).ToList();
        }

        protected LayeredNet(LayeredNet other, int batchSize, int seqLength)
        {
            BatchSize = batchSize;
            SeqLen = seqLength;
            LayersList = other.LayersList.Select(x => x.Clone()).ToList();

            foreach (var layer in LayersList)
            {
                layer.Initialize(batchSize, seqLength);
            }
        }

        private LayeredNet()
        {
        }

        public override int InputSize => InLayer.InputSize;
        public override int OutputSize => OutLayer.OutputSize;

        public override OptimizerBase Optimizer
        {
            get { return _optimizer; }
            set
            {
                _optimizer = value;
            }
        }

        public override int TotalParamCount
        {
            get
            {
                var cnt = 0;
                foreach (var layer in LayersList)
                    cnt += layer.TotalParamCount;
                return cnt;
            }
        }

        public IReadOnlyList<LayerBase> Layers => LayersList;
 
        protected LayerBase OutLayer => LayersList[LayersList.Count - 1];
        protected LayerBase InLayer => LayersList[0];

        public override NeuralNet Clone()
        {
            return new LayeredNet(this);
        }

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

        public override void Optimize()
        {
            foreach (var layer in LayersList)
            {
                layer.Optimize(Optimizer);
            }
        }

        public override double Error(Matrix y, Matrix target)
        {
            return OutLayer.LayerError(y, target);
        }

        public override List<Matrix> BackPropagate(List<Matrix> targets, bool needInputSens = false)
        {  
            List<Matrix> prop = OutLayer.ErrorPropagate(targets);
            if (LayersList.Count < 2)
                return prop;
            for (int i = LayersList.Count - 2; i > 0; i--)
            {
                var layer = LayersList[i];
                prop = layer.BackPropagate(prop, true);
            }
            return InLayer.BackPropagate(prop, needInputSens);
        }

        public override Matrix Step(Matrix input, bool inTraining = false)
        {
            var prop = input;
            foreach (var layer in LayersList)
                prop = layer.Step(prop, inTraining);
            return prop;
        }

        public override void ResetMemory()
        {
            foreach (var layer in LayersList)
                layer.ResetMemory();
        }

        public override void ResetOptimizer()
        {
            foreach (var layer in LayersList)
                layer.ResetOptimizer();
        }

        public override void InitSequence()
        {
            foreach (var layer in LayersList)
                layer.InitSequence();
        }

        public void Dispose()
        {
        }

        public override IReadOnlyList<NeuroWeight> Weights => LayersList.SelectMany(x => x.Weights).ToList();
    }
}