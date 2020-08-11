using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib.Layers
{
    public abstract class Layer
    {
        public int InDim { get; internal set; } = -1;

        public int OutDim { get; protected set; }

        public abstract Tensor ForwardPass(Tensor input, bool training = false);

        public abstract void Compile(IOptimizer optimizer);

        public abstract Tensor BackwardPass(Tensor previousGradient, out Tensor derivativeWeights, out Tensor derivativeBias);
    }
}
