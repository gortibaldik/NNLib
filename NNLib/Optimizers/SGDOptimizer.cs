using System;
using System.Text;
using System.Collections.Generic;

using NNLib.Layers;

namespace NNLib.Optimizers
{
    public class SGDOptimizer : IOptimizer
    {
        private double learningRate;
        private List<Tensor> gradientsWeights = new List<Tensor>();
        private List<Tensor> gradientsBias = new List<Tensor>();

        private bool compiled;

        public SGDOptimizer(double learningRate = 0.01)
        {
            this.learningRate = learningRate;
        }

        public void AddLayer(Layer layer)
        {
            if (compiled == true)
                throw new InvalidOperationException("Optimizer already compiled!");

            var trainable = layer as ITrainable;
            
            if (trainable == null)
            {
                this.gradientsWeights.Add(null);
                this.gradientsBias.Add(null);
            }
            else
            {
                var weights = trainable.GetWeights();
                var bias = trainable.GetBias();

                this.gradientsWeights.Add(new Tensor(weights.BatchSize,weights.Depth, weights.Rows, weights.Columns));
                this.gradientsBias.Add(bias == null ? null : new Tensor(weights.BatchSize, bias.Depth, bias.Rows, bias.Columns));
            }
        }

        public void Compile()
            => compiled = true;


        public void RememberGradient(int index, Tensor gradientWeights, Tensor gradientBias)
        {
            if (gradientsWeights[index] != null)
            {
                gradientsWeights[index] += gradientWeights;
                if (gradientsBias[index] != null)
                    gradientsBias[index] += gradientBias;
            }
        }


        public void CalculateAndUpdateWeights(int sizeOfMiniBatch, List<Layer> layers)
        {
            if (layers.Count != gradientsWeights.Count)
                throw new ArgumentException($"{nameof(layers)} doesn't match {nameof(gradientsWeights)} !");

            for (int i = 0; i < layers.Count; i++)
            {
                var trainable = layers[i] as ITrainable;
                if (trainable != null)
                {
                    var originalWeights = trainable.GetWeights();
                    trainable.SetWeights(originalWeights - ((1.0 * learningRate) / sizeOfMiniBatch) * gradientsWeights[i]);
                    gradientsWeights[i] = gradientsWeights[i].ZeroOut();

                    var originalBias = trainable.GetBias();
                    if (originalBias != null)
                    {
                        trainable.SetBias(originalBias - ((1.0 * learningRate) / sizeOfMiniBatch) * gradientsBias[i]);
                        gradientsBias[i] = gradientsBias[i].ZeroOut();
                    }
                }

            }
        }

        public override string ToString()
        {
            var builder = new StringBuilder();
            foreach (var gw in gradientsWeights)
            {
                builder.Append(gw + "\n");
            }
            return builder.ToString();
        }
    }
}