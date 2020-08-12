using System;
using System.Collections.Generic;
using System.Text;

using NNLib.Layers;
using NNLib.Losses;
using NNLib.Optimizers;

[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("NNLibXUnitTest")]

namespace NNLib
{
    public class NeuralNetwork
    {
        private List<Layer> layers = new List<Layer>();
        private ILossLayer loss = null;
        private IOptimizer optimizer = null;

        private bool compiled = false;
        private int? sizeOfMiniBatch = null;

        public Tensor LastPrediction { get; private set; }

        private void DimCheck(ref int? inDim, int? expected)
        {
            if (inDim != null)
            {
                if (inDim != expected)
                    throw new FormatException("Added layer doesn't fit expected dimensions!");
            }
            else
            {
                inDim = expected;
            }
        }

        public void Add(Layer layer)
        {
            if (compiled)
                compiled = false;

            if (layers.Count == 0)
            {
                if (layer.InDepth == null || layer.InRows == null || layer.InColumns == null)
                    throw new ArgumentException("You must specify the input shape for the first layer !");
            }
            else
            {
                // checking initialized and initializing unitialized dimensions
                int? variable = layer.InDepth;
                DimCheck(ref variable, layers[^1].OutDepth);
                layer.InDepth = variable;

                variable = layer.InRows;
                DimCheck(ref variable, layers[^1].OutRows);
                layer.InRows = variable;

                variable = layer.InColumns;
                DimCheck(ref variable, layers[^1].OutColumns);
                layer.InColumns = variable;
            }

            layers.Add(layer);
        }

        public void Compile(ILossLayer loss, IOptimizer optimizer)
        {
            this.loss = loss == null ? throw new ArgumentException("Loss must be specified!") : loss;
            this.optimizer = optimizer == null ? throw new ArgumentException("Optimizer must be specified!") : optimizer;

            foreach (var layer in layers)
            {
                layer.Compile();
                optimizer.AddLayer(layer);
            }

            compiled = true;
        }

        public Tensor Predict(Tensor input)
        {
            if (!compiled)
                throw new InvalidOperationException("Cannot predict on uncompiled network!");

            if (input.Rows != layers[0].InRows || input.Columns != layers[0].InColumns || input.Depth != layers[0].InDepth)
                throw new ArgumentException("Input dimensions doesn't match first layers inDimensions !");

            Tensor currentOutput = input;
            foreach (var layer in layers)
                currentOutput = layer.ForwardPass(currentOutput);

            return currentOutput;
        }

        public Tensor Forward(Tensor input, Tensor expectedOutput)
        {
            if (input.Rows != layers[0].InRows)
                throw new ArgumentException("Input dimensions doesn't match first layers inDimensions !");

            Tensor currentOutput = input;
            foreach (var layer in layers)
                currentOutput = layer.ForwardPass(currentOutput);

            LastPrediction = currentOutput;
            currentOutput = loss.ForwardPass(currentOutput, expectedOutput);
            return currentOutput;
        }

        public void Backward()
        {
            Tensor currentGradient = loss.BackwardPass();

            for (int i = layers.Count - 1; i >= 0; i--)
            {
                currentGradient = layers[i].BackwardPass(currentGradient, out Tensor gradientWeights, out Tensor gradientBias);
                optimizer.UpdateGradient(i, gradientWeights, gradientBias);
            }

            sizeOfMiniBatch ??= 0;
            sizeOfMiniBatch++;
        }

        public void UpdateWeights()
        {
            optimizer.CalculateAndUpdateWeights((int)sizeOfMiniBatch, layers);
            sizeOfMiniBatch = null;
        }

        public override string ToString()
        {
            var builder = new StringBuilder();
            foreach (var layer in layers)
            {
                builder.Append(layer);
            }
            return builder.ToString();
        }


    }
}
