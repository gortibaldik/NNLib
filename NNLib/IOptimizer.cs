using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib
{
    public interface IOptimizer
    {
        void AddLayer(Matrix weights, Matrix bias);
        void Compile();
        void UpdateGradient(int index, Matrix gradientWeights, Matrix gradientBias);
        (Matrix weights, Matrix bias) CalculateUpdatedWeights(int sizeOfMiniBatch, int index, Matrix originalWeights, Matrix originalBias);
    }
}
