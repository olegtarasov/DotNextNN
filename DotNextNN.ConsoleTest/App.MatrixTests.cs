using System;
using System.Diagnostics;
using System.Linq;
using CLAP;
using DotNextNN.Core;

namespace DotNextNN.ConsoleTest
{
    internal partial class App
    {
        [Verb]
        public void CompareBlasWithLoops()
        {
            const int M = 100;
            const int K = 200;
            const int N = 150;

            const int WarmupCount = 100;
            const int TestCount = 2000;
            
            const float MaxValue = 5;
            const float MinValue = -MaxValue;

            Console.WriteLine("Generating matrices");
            
            // Generate some warmup matrices
            var a_warmup = Enumerable.Range(0, WarmupCount).Select(x => Matrix.RandomMatrix(M, K, MinValue, MaxValue)).ToArray();
            var b_warmup = Enumerable.Range(0, WarmupCount).Select(x => Matrix.RandomMatrix(K, N, MinValue, MaxValue)).ToArray();
            var a_slow_warmup = a_warmup.Select(SlowMatrix.FromMatrix).ToArray();
            var b_slow_warmup = b_warmup.Select(SlowMatrix.FromMatrix).ToArray();

            // Generate MKL matrices
            var a = Enumerable.Range(0, TestCount).Select(x => Matrix.RandomMatrix(M, K, MinValue, MaxValue)).ToArray();
            var b = Enumerable.Range(0, TestCount).Select(x => Matrix.RandomMatrix(K, N, MinValue, MaxValue)).ToArray();

            // Copy them as slow matrices
            var a_slow = a.Select(SlowMatrix.FromMatrix).ToArray();
            var b_slow = b.Select(SlowMatrix.FromMatrix).ToArray();

            Console.WriteLine("Warming up OpenBLAS and loops");

            // Warm up
            double sum = 0.0d;
            for (int i = 0; i < WarmupCount; i++)
            {
                var result = a_warmup[i] * b_warmup[i];
                sum += result[0, 0];
            }

            Console.WriteLine($"OpenBLAS warmup complete, useless sum: {sum}");

            sum = 0.0d;
            for (int i = 0; i < WarmupCount; i++)
            {
                var result = SlowMatrix.DotProduct(a_slow_warmup[i], b_slow_warmup[i]);
                sum += result[0][0];
            }

            Console.WriteLine($"Loops warmup complete, useless sum: {sum}");

            Console.WriteLine("Testing OpenBLAS");

            sum = 0.0d;
            var watch = new Stopwatch();
            watch.Start();
            for (int i = 0; i < TestCount; i++)
            {
                var result = a[i] * b[i];
                sum += result[0, 0];
            }
            watch.Stop();

            Console.WriteLine($"OpenBLAS time: {watch.Elapsed}, useless sum: {sum}");

            Console.WriteLine("Testing loops");
            sum = 0.0d;
            watch.Restart();
            for (int i = 0; i < TestCount; i++)
            {
                var result = SlowMatrix.DotProduct(a_slow[i], b_slow[i]);
                sum += result[0][0];
            }
            watch.Stop();

            Console.WriteLine($"Loops time: {watch.Elapsed}, useless sum: {sum}");
        }
    }
}