using System;
using System.IO;
using System.Text;

namespace DotNextNN.Core.Neural
{
	public class NeuroWeight
    {
        /// <summary>
        /// Weight matrix
        /// </summary>
		public Matrix Weight { get; private set; }

        /// <summary>
        /// Current gradient matrix of Weight matrix
        /// </summary>
		public Matrix Gradient { get; private set; }

        /// <summary>
        /// Grad1 avg cache
        /// </summary>
		public Matrix Cache1 { get; private set; }

        /// <summary>
        /// Grad2 avg cache
        /// </summary>
        public Matrix Cache2 { get; private set; }

        /// <summary>
        /// Timestep
        /// </summary>
        public int Timestep { get; set; } = 0;

        public NeuroWeight()
		{
            Guid.NewGuid();
		}

		public NeuroWeight(Matrix weight) : this()
		{
            Guid.NewGuid();
			Weight = weight.Clone();
			Gradient = new Matrix(weight.Rows, weight.Cols);
            Cache1 = new Matrix(weight.Rows, weight.Cols);
            Cache2 = new Matrix(weight.Rows, weight.Cols);
            Timestep = 0;
		}

		private NeuroWeight(NeuroWeight other)
		{
		    Weight = other.Weight.Clone();
			Gradient = other.Gradient.Clone();
			Cache1 = other.Cache1.Clone();
            Cache2 = other.Cache2.Clone();
		    Timestep = other.Timestep;
		}

		public NeuroWeight Clone()
		{
			return new NeuroWeight(this);
		}

		public void ClearGrad()
		{
			Gradient.Clear();
		}

		public void ClearCache()
		{
			Cache1.Clear();
            Cache2.Clear();
		    Timestep = 0;
		}
	}
}