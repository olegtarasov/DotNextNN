using System;
using System.Collections.Generic;
using System.IO;
using DotNextNN.Core.Neural.Initializers;
using DotNextNN.Core.Optimizers;

namespace DotNextNN.Core.Neural.Layers
{
    public enum AffineActivation
    {
        None,
        Sigmoid
    }

    public class AffineLayer : LayerBase
    {
        private readonly LayerBase _activationLayer;
        private readonly AffineActivation _activationType;
        private readonly LinearLayer _linearLayer;


        public AffineLayer(int xSize, int ySize, AffineActivation activation)
        {
            _activationType = activation;
            _linearLayer = new LinearLayer(xSize, ySize);
            _activationLayer = GetAffineActivationLayer(activation, ySize);
        }

        public AffineLayer(int xSize, int ySize, AffineActivation activation, IMatrixInitializer matrixInitializer)
        {
            _activationType = activation;
            _linearLayer = new LinearLayer(xSize, ySize, matrixInitializer);
            _activationLayer = GetAffineActivationLayer(activation, ySize);
        }

        public AffineLayer(AffineLayer other) : base(other)
        {
            _activationType = other._activationType;
            _linearLayer = (LinearLayer)other._linearLayer.Clone();
            _activationLayer = other._activationLayer.Clone();
        }

        public override int InputSize => _linearLayer.InputSize;

        public override int OutputSize => _activationLayer.OutputSize;
        public override int TotalParamCount => _linearLayer.TotalParamCount + _activationLayer.TotalParamCount;

        public override Matrix BackPropagate(Matrix outSens, bool needInputSens = true, bool clearGrad = true)
        {
            var activationSens = _activationLayer.BackPropagate(outSens, needInputSens, clearGrad);
            return _linearLayer.BackPropagate(activationSens, needInputSens, clearGrad);
        }

        public override void ClampGrads(float limit)
        {
            _linearLayer.ClampGrads(limit);
            _activationLayer.ClampGrads(limit);
        }

        public override LayerBase Clone()
        {
            return new AffineLayer(this);
        }

        public override Matrix ErrorPropagate(Matrix targets)
        {
            return BackPropagate(base.ErrorPropagate(targets));
        }

        public override void InitSequence()
        {
            Input.Clear();
            Output.Clear();
            _linearLayer.InitSequence();
            _activationLayer.InitSequence();
        }

        public override void Optimize(OptimizerBase optimizer)
        {
            _linearLayer.Optimize(optimizer);
            _activationLayer.Optimize(optimizer);
        }

        public override void ResetMemory()
        {
            _linearLayer.ResetMemory();
            _activationLayer.ResetMemory();
        }

        public override void ResetOptimizer()
        {
            _linearLayer.ResetOptimizer();
            _activationLayer.ResetOptimizer();
        }

        public override Matrix Step(Matrix input, bool inTraining = false)
        {
            Input = input;
            var output = _activationLayer.Step(_linearLayer.Step(input, inTraining), inTraining);
            Output = output;
            return output;
        }

        protected override void Initialize()
        {
            _linearLayer.Initialize(BatchSize, SeqLen);
            _activationLayer.Initialize(BatchSize, SeqLen);
        }

        private LayerBase GetAffineActivationLayer(AffineActivation activation, int ySize)
        {
            switch (activation)
            {
                case AffineActivation.Sigmoid:
                    return new SigmoidLayer(ySize);
                default:
                    throw new ArgumentOutOfRangeException(nameof(activation), activation, null);
            }
        }

        private LayerBase LoadAffineActivationLayer(AffineActivation activation, BinaryReader reader)
        {
            switch (activation)
            {
                case AffineActivation.Sigmoid:
                    return new SigmoidLayer(reader);
                default:
                    throw new ArgumentOutOfRangeException(nameof(activation), activation, null);
            }
        }

        public override IReadOnlyList<NeuroWeight> Weights => _linearLayer.Weights;
        public override void ClearGradients()
        {
            _linearLayer.ClearGradients();
            _activationLayer.ClearGradients();
        }
    }
}