using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Linq;
//using Accord.Imaging.Converters;
//using Accord.Math;
using CLAP;
using DotNextNN.Core;
using DotNextNN.Core.Neural;
using DotNextNN.Core.Neural.Layers;
using DotNextNN.Core.Optimizers;

namespace DotNextNN.ConsoleTest
{
    internal partial class App
    {
        [Verb]
        public void TrainMNIST(string path, int epochs = 5)
        {
            const int batchSize = 128;
            const int hSize = 20;

            var errFile = new FileStream($"ERR_{DateTime.Now:dd_MM_yy-HH_mm_ss}.txt", FileMode.Create);
            var testErrFile = new FileStream($"ERR_TEST_{DateTime.Now:dd_MM_yy-HH_mm_ss}.txt", FileMode.Create);
            var errWriter = new StreamWriter(errFile);
            var testErrWriter = new StreamWriter(testErrFile);

            var trainSet = MnistDataSet.Load(Path.Combine(path, "train-images-idx3-ubyte"), Path.Combine(path, "train-labels-idx1-ubyte"));
            var testSet = MnistDataSet.Load(Path.Combine(path, "t10k-images-idx3-ubyte"), Path.Combine(path, "t10k-labels-idx1-ubyte"));

            var optimizer = new AdamOptimizer();
            var network = new LayeredNet(batchSize,
                new LinearLayer(trainSet.InputSize, hSize),
                new SigmoidLayer(hSize),
                new LinearLayer(hSize, trainSet.TargetSize),
                new SoftMaxLayer(trainSet.TargetSize));

            network.Optimizer = optimizer;
            trainSet.BatchSize = batchSize;
            testSet.BatchSize = batchSize;

            int iter = 0;
            double epoch = 0.0;
            
            var watch = new Stopwatch();

            const int FILTER_SIZE = 100;
            var filter = new List<double>(FILTER_SIZE);
            var globErrList = new List<double>();

            watch.Restart();

            while (true)
            {
                var sequence = trainSet.GetNextSample();
                double error = network.Train(sequence.Input, sequence.Target);

                network.Optimize();

                epoch = (double)(iter * batchSize) / trainSet.SampleCount;

                filter.Add(error);
                if (filter.Count > FILTER_SIZE)
                    filter.RemoveAt(0);

                var err = filter.Sum() / FILTER_SIZE;
                if (filter.Count == FILTER_SIZE)
                    globErrList.Add(err);
                if (filter.Count < FILTER_SIZE)
                    err = error;

                iter++;


                if (iter % 10 == 0)
                {
                    watch.Stop();
                    Console.WriteLine($"Epoch #{epoch.ToString("F3", NumberFormatInfo.InvariantInfo)} - iter #{iter}:");
                    Console.WriteLine("---------");
                    Console.WriteLine("\tError:\t\t{0:0.0000}", err);
                    Console.WriteLine("\tDuration:\t{0:0.0000}s", watch.Elapsed.TotalSeconds);
                    Console.WriteLine("---------\n");

                    Test(network, testSet, testErrWriter);

                    watch.Restart();
                }

                if (iter % 500 == 0)
                {
                    foreach (var e in globErrList)
                    {
                        errWriter.WriteLine(e);
                    }

                    errFile.Flush(true);

                    globErrList.Clear();

                }

                if (epoch > epochs)
                {
                    Console.WriteLine($"{epochs} epochs reached, finishing training");
                    VisualTest(network, testSet);

                    Console.ReadKey();
                    return;
                }
            }
        }

        private void Test(LayeredNet network, MnistDataSet dataSet, StreamWriter writer)
        {
            var samples = dataSet.GetNextSample();

            network.Test(samples.Input, samples.Target, out var error);

            writer.WriteLine(error);
            writer.Flush();
        }

        private void VisualTest(LayeredNet network, MnistDataSet dataSet)
        {
            dataSet.BatchSize = 10;
            var samples = dataSet.GetNextSample();
            network.BatchSize = 10;

            //var converter = new MatrixToImage(1.0, 0);

            using (var writer = new StreamWriter("VisualTest.txt"))
            {

                writer.WriteLine("Predicted,Actual");

                var result = network.Step(samples.Input);
                var predicted = SoftMaxLayer.SoftMaxChoice(result);
                var target = samples.Target;

                for (int i = 0; i < 10; i++)
                {
                    int targetClass = -1;

                    for (int classIdx = 0; classIdx < samples.Target.Rows; classIdx++)
                    {
                        if ((int)target[classIdx, i] == 1)
                        {
                            targetClass = classIdx;
                            break;
                        }
                    }

                    if (targetClass < 0)
                    {
                        throw new InvalidOperationException("Target vector doesn't contain a positive result!");
                    }

                    writer.WriteLine($"{predicted[i]},{targetClass}");

                    var input = samples.Input;
                    int offset = input.Rows * i;
                    var imageArray = new float[input.Rows];
                    Array.Copy((float[])input, offset, imageArray, 0, input.Rows);

                    //converter.Convert(imageArray.Reshape(28, 28), out Bitmap bitmap);
                    //bitmap.Save($"{i}.bmp", ImageFormat.Bmp);
                }
            }
        }
    }
}