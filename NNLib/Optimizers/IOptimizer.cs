using NNLib.Layers;
using System.Collections.Generic;

namespace NNLib.Optimizers
{
    public interface IOptimizer
    {
        /// <summary>
        /// Adds layers to the layers trained by the optimizer. If the layer implements 
        /// ITrainable interface, the optimizer trains it, otherwise it only keeps track
        /// that there exists such a layer and doesn't try to do anything with it.
        /// </summary>
        void AddLayer(Layer layer);

        /// <summary>
        /// Prepares the optimizer for the usage.
        /// </summary>
        void Compile();

        /// <summary>
        /// Adds gradientWeights and gradientBias to the currently remembered gradients of
        /// the layer on specified index.
        /// </summary>
        void RememberGradient(int index, Tensor gradientWeights, Tensor gradientBias);

        /// <summary>
        /// Based on the remembered gradients and the algorithm of the optimizer, the
        /// weights are updated.
        /// </summary>
        void CalculateAndUpdateWeights(int sizeOfMiniBatch, List<Layer> layers);
    }
}
