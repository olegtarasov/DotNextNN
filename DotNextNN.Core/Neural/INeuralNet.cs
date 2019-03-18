namespace DotNextNN.Core.Neural
{
    public interface INeuralNet
    {
        void Optimize();
        int InputSize { get; }
        int OutputSize { get; }
        int TotalParamCount { get; }
    }
}