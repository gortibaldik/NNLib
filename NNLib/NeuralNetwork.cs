using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib
{
    public class NeuralNetwork
    {
        private List<Layer> layers = new List<Layer>();
        private LossLayer loss = null;
        private IOptimizer optimizer = null;

        private bool compiled = false;
        private int? sizeOfMiniBatch = null;

        public void Add(Layer layer)
        {
            if (compiled)
                compiled = false;

            if (layers.Count == 0 && layer.InDim == -1)
                throw new ArgumentException("No Input dimensions specified for the first layer !");

            else if (layers.Count != 0 && layer.InDim != layers[^1].OutDim)
                throw new ArgumentException("Invalid input dimension of next layer!");
            
            else if (layer.InDim == -1)
                layer.InDim = layers[^1].OutDim;


            layers.Add(layer);
        }

        public void Compile(LossLayer loss, IOptimizer optimizer)
        {
            this.loss = loss == null ? throw new ArgumentException("Loss must be specified!") : loss;
            this.optimizer = optimizer == null ? throw new ArgumentException("Optimizer must be specified!") : optimizer;
            
            foreach (var layer in layers)
                layer.Compile(optimizer);

            compiled = true;
        }

        public Matrix Predict(Matrix input)
        {
            if (!compiled)
                throw new InvalidOperationException("Cannot predict on uncompiled network!");

            if (input.Rows != layers[0].InDim)
                throw new ArgumentException("Input dimensions doesn't match first layers inDimensions !");

            Matrix currentOutput = input;
            foreach (var layer in layers)
                currentOutput = layer.ForwardPass(currentOutput);

            return currentOutput;
        }

        public Matrix Forward(Matrix input, Matrix expectedOutput)
        {
            if (input.Rows != layers[0].InDim)
                throw new ArgumentException("Input dimensions doesn't match first layers inDimensions !");

            Matrix currentOutput = input;
            foreach (var layer in layers)
                currentOutput = layer.ForwardPass(currentOutput);

            currentOutput = loss.ForwardPass(currentOutput, expectedOutput);
            return currentOutput;
        }

        public void Backward()
        {
            Matrix currentGradient = loss.BackwardPass();

            for (int i = layers.Count - 1; i >= 0; i--)
            {
                currentGradient = layers[i].BackwardPass(currentGradient, out Matrix gradientWeights, out Matrix gradientBias);
                optimizer.UpdateGradient(i, gradientWeights, gradientBias);
            }

            sizeOfMiniBatch ??= 0;
            sizeOfMiniBatch++;
        }

        public void UpdateWeights()
        {
            for (int i = 0; i < layers.Count; i++)
            {
                var trainable = layers[i] as TrainableLayer;
                if (trainable != null)
                {
                    (var newWeights, var newBias) = optimizer.CalculateUpdatedWeights((int)sizeOfMiniBatch, i, trainable.GetWeights(), trainable.GetBias());
                    trainable.SetWeights(newWeights);
                    trainable.SetBias(newBias);
                }
            }
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
