using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib.Activations
{
    class ActivationFactory
    {
        public IActivationLayer Create(string activation)
        {
            switch(activation)
            {
                case nameof(ReLU): return new ReLU();
                case nameof(Softmax): return new Softmax();
                default: return new LinearActivation();
            }
        }
    }
}
