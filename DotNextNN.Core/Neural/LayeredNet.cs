﻿using System;
using System.Collections.Generic;
using System.IO;
using DotNextNN.Core.Neural.Layers;
using DotNextNN.Core.Optimizers;


namespace DotNextNN.Core.Neural
{
    public class LayeredNet
    {
        private int _batchSize;

        private readonly List<LayerBase> _layersList = new List<LayerBase>();

        public LayeredNet(int batchSize, params LayerBase[] layers)
        {
            if (layers.Length == 0) throw new ArgumentException("Value cannot be an empty collection.", nameof(layers));

            _batchSize = batchSize;

            for (int i = 0; i < layers.Length; i++)
            {
                var layer = layers[i];
                _layersList.Add(layer);
                layer.Initialize(batchSize);
                if (i == 0)
                    continue;
               if (layers[i-1].OutputSize != layer.InputSize)
                    throw new ArgumentException($"Dimension of layer #{i} and #{i+1} does not agree ({layers[i-1].OutputSize}!={layer.InputSize})!");
            }
            
        }

        public int BatchSize
        {
            get => _batchSize;
            set
            {
                _batchSize = value;
                foreach (var layer in _layersList)
                {
                    layer.Initialize(value);
                }
            }
        }


        private LayerBase OutLayer => _layersList[_layersList.Count - 1];
        private LayerBase InLayer => _layersList[0];

        public double Error(Matrix y, Matrix target)
        {
            return OutLayer.LayerError(y, target);
        }

        public Matrix Step(Matrix input, bool inTraining = false)
        {
            var prop = input;
            foreach (var layer in _layersList)
                prop = layer.Step(prop, inTraining);
            return prop;
        }
    }
}