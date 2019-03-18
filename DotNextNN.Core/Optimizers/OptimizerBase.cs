using System;
using DotNextNN.Core.Neural;

namespace DotNextNN.Core.Optimizers
{
	public abstract class OptimizerBase
	{
	    protected OptimizerBase(float learningRate)
	    {
	        LearningRate = learningRate;
	    }

	    public float LearningRate { get; set; }

        public abstract void Optimize(NeuroWeight weight);
	}
}