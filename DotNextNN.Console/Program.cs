using System;
using System.Diagnostics;
using System.Linq;
using CLAP;
using DotNextNN.Core;

namespace DotNextNN.ConsoleTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Parser.Run(args, new App());
        }
    }

    class App
    {
        [Verb]
        public void CompareMklWithLoops()
        {
            // Generate some warmup matrices
            var a_warmup = Enumerable.Range(0, 100).Select(x => Matrix.RandomMatrix(100, 200, -5, 5)).ToArray();
            var b_warmup = Enumerable.Range(0, 100).Select(x => Matrix.RandomMatrix(200, 150, -5, 5)).ToArray();
            var a_slow_warmup = a_warmup.Select(SlowMatrix.FromMatrix).ToArray();
            var b_slow_warmup = b_warmup.Select(SlowMatrix.FromMatrix).ToArray();

            // Generate MKL matrices
            var a = Enumerable.Range(0, 2000).Select(x => Matrix.RandomMatrix(100, 200, -5, 5)).ToArray();
            var b = Enumerable.Range(0, 2000).Select(x => Matrix.RandomMatrix(200, 150, -5, 5)).ToArray();

            // Copy them as slow matrices
            var a_slow = a.Select(SlowMatrix.FromMatrix).ToArray();
            var b_slow = b.Select(SlowMatrix.FromMatrix).ToArray();

            Console.WriteLine("Warming up MKL and loops");

            // Warm up
            double sum = 0.0d;
            for (int i = 0; i < 100; i++)
            {
                var result = a_warmup[i] * b_warmup[i];
                sum += result[0, 0];
            }

            Console.WriteLine($"MKL warmup complete, useless sum: {sum}");

            sum = 0.0d;
            for (int i = 0; i < 100; i++)
            {
                var result = SlowMatrix.DotProduct(a_slow_warmup[i], b_slow_warmup[i]);
                sum += result[0][0];
            }

            Console.WriteLine($"Loops warmup complete, useless sum: {sum}");

            Console.WriteLine("Testing MKL");

            sum = 0.0d;
            var watch = new Stopwatch();
            watch.Start();
            for (int i = 0; i < 2000; i++)
            {
                var result = a[i] * b[i];
                sum += result[0, 0];
            }
            watch.Stop();

            Console.WriteLine($"MKL time: {watch.Elapsed}, useless sum: {sum}");

            Console.WriteLine("Testing loops");
            sum = 0.0d;
            watch.Restart();
            for (int i = 0; i < 2000; i++)
            {
                var result = SlowMatrix.DotProduct(a_slow[i], b_slow[i]);
                sum += result[0][0];
            }
            watch.Stop();

            Console.WriteLine($"Loops time: {watch.Elapsed}, useless sum: {sum}");
        }
    }
}
