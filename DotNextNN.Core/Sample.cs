using System.IO;

namespace DotNextNN.Core
{
    public class Sample
    {
        public Sample()
        {
        }

        public Sample(Matrix input, Matrix target)
        {
            Input = input;
            Target = target;
        }

        public Sample(int inputSize, int targetSize, int batchSize)
        {
            Input = new Matrix(inputSize, batchSize);
            Target = new Matrix(targetSize, batchSize);
        }

        public Matrix Input { get; set; }
        public Matrix Target { get; set; }
    }
}