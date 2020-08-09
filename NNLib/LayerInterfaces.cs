using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib
{
    public abstract class Layer
    {
        public int InDim { get; internal set; } = -1;

        public int OutDim { get; protected set; }

        public abstract Tensor ForwardPass(Tensor input, bool training = false);

        public abstract void Compile(IOptimizer optimizer);

        public abstract Tensor BackwardPass(Tensor previousGradient, out Tensor derivativeWeights, out Tensor derivativeBias);
    }

    interface TrainableLayer
    {
        Tensor GetWeights();
        Tensor GetBias();
        void SetWeights(Tensor weights);
        void SetBias(Tensor bias);
    }

    public interface ActivationLayer
    {
        Tensor ForwardPass(Tensor input);

        Tensor BackwardPass(Tensor previousGradient);
    }

    public interface LossLayer
    {
        Tensor ForwardPass(Tensor neuralOutput, Tensor expectedOutput);

        Tensor BackwardPass();
    }
}
