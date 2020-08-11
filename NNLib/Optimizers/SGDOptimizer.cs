using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib.Optimizers
{
    public class SGDOptimizer : IOptimizer
    {
        private double learningRate;
        private List<Tensor> gradientsWeights = new List<Tensor>();
        private List<Tensor> gradientsBias = new List<Tensor>();

        private bool compiled;
        private int numberOfLayers;

        public SGDOptimizer(double learningRate)
        {
            this.learningRate = learningRate;
        }

        public void AddLayer(Tensor weights, Tensor bias)
        {
            if (compiled == true)
                throw new InvalidOperationException("Optimizer already compiled!");

            // needs to be tested
            if (weights == null)
            {
                this.gradientsWeights.Add(null);
                this.gradientsBias.Add(null);
            }
            else
            {
                this.gradientsWeights.Add(new Tensor(weights.Depth, weights.Rows, weights.Columns));
                this.gradientsBias.Add(bias == null ? null : new Tensor(bias.Depth, bias.Rows, bias.Columns));
            }
        }

        public void Compile()
        {
            compiled = true;
            numberOfLayers = gradientsWeights.Count;
        }

        // needs to be tested
        public void UpdateGradient(int index, Tensor gradientWeights, Tensor gradientBias)
        {
            if (gradientsWeights[index] != null)
            {
                gradientsWeights[index] += gradientWeights;
                if (gradientsBias[index] != null)
                    gradientsBias[index] += gradientBias;
            }
        }

        // needs to be tested
        public (Tensor weights, Tensor bias) CalculateUpdatedWeights(int sizeOfMiniBatch, int index, Tensor originalWeights, Tensor originalBias)
        {
            if (gradientsWeights[index] == null)
                throw new InvalidOperationException("Cannot calculate updated weights of non-trainable layer !");

            var newWeights = originalWeights - ((1.0 * learningRate) / sizeOfMiniBatch) * gradientsWeights[index];

            Tensor newBias = null;
            if (originalBias != null)
                newBias = originalBias - ((1.0 * learningRate) / sizeOfMiniBatch) * gradientsBias[index];

            return (newWeights, newBias);
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
