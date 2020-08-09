using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib
{
    public class SGDOptimizer : IOptimizer
    {
        private double learningRate;
        private List<Matrix> gradientsWeights = new List<Matrix>();
        private List<Matrix> gradientsBias = new List<Matrix>();

        private bool compiled;
        private int numberOfLayers;

        public SGDOptimizer(double learningRate)
        {
            this.learningRate = learningRate;
        }

        public void AddLayer(Matrix weights, Matrix bias)
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
                this.gradientsWeights.Add(new Matrix(weights.Rows, weights.Columns));
                this.gradientsBias.Add(bias == null ? null : new Matrix(bias.Rows, bias.Columns));
            }
        }

        public void Compile()
        {
            compiled = true;
            numberOfLayers = gradientsWeights.Count;
        }

        // needs to be tested
        public void UpdateGradient(int index, Matrix gradientWeights, Matrix gradientBias)
        {
            if (gradientsWeights[index] != null)
            {
                gradientsWeights[index] += gradientWeights;
                if (gradientsBias[index] != null)
                    gradientsBias[index] += gradientBias;
            }
        }

        // needs to be tested
        public (Matrix weights, Matrix bias) CalculateUpdatedWeights(int sizeOfMiniBatch, int index, Matrix originalWeights, Matrix originalBias)
        {
            if (gradientsWeights[index] == null)
                throw new InvalidOperationException("Cannot calculate updated weights of non-trainable layer !");

            var newWeights = originalWeights - ((1.0 * learningRate) / sizeOfMiniBatch) * gradientsWeights[index];

            Matrix newBias = null;
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
