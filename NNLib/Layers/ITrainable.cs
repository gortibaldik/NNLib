using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib.Layers
{
    /// <summary>
    /// Interface representing a layer with trainable heights and biases
    /// </summary>
    interface ITrainable
    {
        Tensor GetWeights();
        Tensor GetBias();
        void SetWeights(Tensor weights);
        void SetBias(Tensor bias);
    }
}
