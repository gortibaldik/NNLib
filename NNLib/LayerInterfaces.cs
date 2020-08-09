using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib
{
    public abstract class Layer
    {
        public int InDim { get; internal set; } = -1;

        public int OutDim { get; protected set; }

        public abstract Matrix ForwardPass(Matrix input, bool training = false);

        public abstract void Compile(IOptimizer optimizer);

        public abstract Matrix BackwardPass(Matrix previousGradient, out Matrix derivativeWeights, out Matrix derivativeBias);
    }

    interface TrainableLayer
    {
        Matrix GetWeights();
        Matrix GetBias();
        void SetWeights(Matrix weights);
        void SetBias(Matrix bias);
    }

    public interface ActivationLayer
    {
        Matrix ForwardPass(Matrix input);

        Matrix BackwardPass(Matrix previousGradient);
    }

    public interface LossLayer
    {
        Matrix ForwardPass(Matrix neuralOutput, Matrix expectedOutput);

        Matrix BackwardPass();
    }
}
