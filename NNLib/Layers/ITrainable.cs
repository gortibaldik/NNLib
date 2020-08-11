using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib.Layers
{
    interface ITrainable
    {
        Tensor GetWeights();
        Tensor GetBias();
        void SetWeights(Tensor weights);
        void SetBias(Tensor bias);
    }
}
