using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Linq;
using Accord.Imaging.Converters;
using Accord.Math;
using CLAP;
using DotNextNN.Core;
using DotNextNN.Core.Neural;
using DotNextNN.Core.Neural.Layers;

namespace DotNextNN.ConsoleTest
{
    internal partial class App
    {
        [Verb]
        public void TrainMNIST(string path)
        {
            const int batchSize = 128;
            const int hSize = 20;

            var errFile = new FileStream($"ERR_{DateTime.Now:dd_MM_yy-HH_mm_ss}.txt", FileMode.Create);
            var testErrFile = new FileStream($"ERR_TEST_{DateTime.Now:dd_MM_yy-HH_mm_ss}.txt", FileMode.Create);
            var errWriter = new StreamWriter(errFile);
            var testErrWriter = new StreamWriter(testErrFile);

            var trainSet = MnistDataSet.Load(Path.Combine(path, "train-images-idx3-ubyte"), Path.Combine(path, "train-labels-idx1-ubyte"));
            var testSet = MnistDataSet.Load(Path.Combine(path, "t10k-images-idx3-ubyte"), Path.Combine(path, "t10k-labels-idx1-ubyte"));

            var network = new LayeredNet(batchSize,
                new LinearLayer(trainSet.InputSize, hSize),
                new SigmoidLayer(hSize),
                new LinearLayer(hSize, trainSet.TargetSize),
                new SoftMaxLayer(trainSet.TargetSize));

            trainSet.BatchSize = batchSize;
            testSet.BatchSize = batchSize;

            int iter = 0;
            double epoch = 0.0;
            
            var watch = new Stopwatch();

            VisualTest(network, testSet);
        }

        private void VisualTest(LayeredNet network, MnistDataSet dataSet)
        {
            dataSet.BatchSize = 10;
            var samples = dataSet.GetNextSample();
            network.BatchSize = 10;

            var converter = new MatrixToImage(1.0, 0);

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

                    converter.Convert(imageArray.Reshape(28, 28), out Bitmap bitmap);
                    bitmap.Save($"{i}.bmp", ImageFormat.Bmp);
                }
            }
        }
    }
}