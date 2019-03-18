using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DotNextNN.Core.Optimizers;

namespace DotNextNN.Core.Neural
{
    public abstract class NeuralNet : INeuralNet
    {
        public virtual OptimizerBase Optimizer { get; set; }

		public abstract NeuralNet Clone();

		public abstract void Optimize();

        public abstract List<Matrix> BackPropagate(List<Matrix> targets, bool needInputSense = false);

        public abstract double Error(Matrix y, Matrix target);

        public abstract Matrix Step(Matrix input, bool inTraining = false);
        
		public abstract void ResetMemory();
		public abstract void ResetOptimizer();
		public abstract void InitSequence();

        public abstract int InputSize { get; }
        public abstract int OutputSize { get; }
		public abstract int TotalParamCount { get; }

        public abstract IReadOnlyList<NeuroWeight> Weights { get; }

        protected NeuralNet()
        {
        }

		public virtual List<Matrix> ProcessSequence(List<Matrix> inputs)
        {
            var yList = new List<Matrix>(inputs.Count);
            foreach (var input in inputs)
            {
                var y = Step(input);
                yList.Add(y);
            }
            return yList;
        }

        public virtual List<Matrix> TestSequence(List<Matrix> inputs, List<Matrix> targets, out List<double> errors)
        {
            var yList = new List<Matrix>(inputs.Count);
            errors=new List<double>(inputs.Count);

            for (int i = 0; i < inputs.Count; i++)
            {
                var input = inputs[i];
                var target = targets[i];
                var y = Step(input);
                var e = Error(y, target);
                errors.Add(e);
                yList.Add(y);
            }
            return yList;
        }

	    public virtual double TrainSequence(List<Matrix> inputs, List<Matrix> targets)
	    {
            if (inputs.Count != targets.Count || targets.Count == 0)
                throw new Exception("Not enough targets or inputs provided!");

            var sequenceLen = inputs.Count;
            InitSequence();
            var error = new List<double>(sequenceLen);
            for (int i = 0; i < inputs.Count; i++)
            {
                var target = targets[i];
                var input = inputs[i];
                var y = Step(input, true);
                error.Add(Error(y, target));
            }
            BackPropagate(targets);
            var totalErr = error.Sum() / error.Count;
            return totalErr;
        }
    }
}