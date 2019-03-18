using System.Collections.Generic;

namespace DotNextNN.Core
{
    public class TrainingSequence
    {
        public TrainingSequence(List<Matrix> inputs, List<Matrix> targets)
        {
            Inputs = inputs;
            Targets = targets;
        }

        public List<Matrix> Inputs { get; set; }
        public List<Matrix> Targets { get; set; }
    }
}