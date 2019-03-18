using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using DotNextNN.Core;
using Retia.RandomGenerator;

namespace DotNextNN.ConsoleTest
{
    public class MnistDataSet
    {
        private readonly List<MnistImage> _images;

        private MnistDataSet(List<MnistImage> images)
        {
            _images = images;
        }

        public static MnistDataSet Load(string imagesPath, string labelsPath)
        {
            var images = new List<float[]>();
            var result = new List<MnistImage>();

            // Load images
            using (var stream = new FileStream(imagesPath, FileMode.Open, FileAccess.Read))
            using (var reader = new BinaryReader(stream))
            {
                int magic = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                if (magic != 2051)
                {
                    throw new InvalidOperationException("MNIST image magic error");
                }

                int imageCount = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                int rows = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                int cols = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                int len = rows * cols;

                for (int img = 0; img < imageCount; img++)
                {
                    var data = new float[len];
                    for (int pt = 0; pt < len; pt++)
                    {
                        data[pt] = reader.ReadByte();
                    }

                    images.Add(data);
                }
            }

            // Load labels
            using (var stream = new FileStream(labelsPath, FileMode.Open, FileAccess.Read))
            using (var reader = new BinaryReader(stream))
            {
                int magic = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                if (magic != 2049)
                {
                    throw new InvalidOperationException("MNIST label magic error");
                }

                int labelCount = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                if (labelCount != images.Count)
                {
                    throw new InvalidOperationException("Label count is not equal to image count!");
                }

                for (int i = 0; i < labelCount; i++)
                {
                    result.Add(new MnistImage(images[i], reader.ReadByte()));
                }
            }

            return new MnistDataSet(result);
        }

        public Sample GetNextSample()
        {
            if (BatchSize <= 0) throw new InvalidOperationException("Set batch size!");

            var gen = SafeRandom.Generator;
            var images = new float[BatchSize][];
            var labels = new float[BatchSize][];
            for (int i = 0; i < BatchSize; i++)
            {
                int idx = gen.Next(_images.Count);
                images[i] = _images[idx].Data;
                labels[i] = new float[10];
                labels[i][_images[idx].Label] = 1.0f;
            }

            return new Sample(
                Matrix.FromColumnArrays(images),
                Matrix.FromColumnArrays(labels));
        }

        public int SampleCount => _images.Count;
        public int InputSize => _images[0].Data.Length;
        public int TargetSize => 10;
        public int BatchSize { get; set; }
    }

    public class MnistImage
    {
        public readonly float[] Data;
        public readonly byte Label;

        public MnistImage(float[] data, byte label)
        {
            Data = data;
            Label = label;
        }
    }
}