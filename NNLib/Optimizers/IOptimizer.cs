using NNLib.Layers;
using System.Collections.Generic;

namespace NNLib.Optimizers
{
    public interface IOptimizer
    {
        void AddLayer(Layer layer);
        void Compile();
        void UpdateGradient(int index, Tensor gradientWeights, Tensor gradientBias);
        void CalculateAndUpdateWeights(int sizeOfMiniBatch, List<Layer> layers);
    }
}
