using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib.Optimizers
{
    public interface IOptimizer
    {
        void AddLayer(Tensor weights, Tensor bias);
        void Compile();
        void UpdateGradient(int index, Tensor gradientWeights, Tensor gradientBias);
        (Tensor weights, Tensor bias) CalculateUpdatedWeights(int sizeOfMiniBatch, int index, Tensor originalWeights, Tensor originalBias);
    }
}
