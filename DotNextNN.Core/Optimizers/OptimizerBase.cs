using System;
using DotNextNN.Core.Neural;

namespace DotNextNN.Core.Optimizers
{
	public abstract class OptimizerBase : IOptimizer
	{
	    protected IntPtr GpuOptimizerPtr = IntPtr.Zero;

        protected OptimizerBase(float learningRate)
	    {
	        LearningRate = learningRate;
	    }

	    protected OptimizerBase(OptimizerBase other)
	    {
	        LearningRate = other.LearningRate;
	    }

	    public float LearningRate { get; set; }

        public abstract void Optimize(NeuroWeight weight);
	}
}