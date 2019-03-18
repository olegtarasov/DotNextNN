using System;
using System.Collections.Generic;
using System.Linq;

namespace DotNextNN.Core.Neural
{
	public static class NeuralExtensions
	{
		public static List<Matrix> Clone(this List<Matrix> list)
        {
			return list.Select(x => x.Clone()).ToList();
		}
	}
}