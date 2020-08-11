using System;

namespace NNLib.Activations
{
    public class ReLU : IActivationLayer
    {
        private Tensor lastInput;
        public Tensor ForwardPass(Tensor input)
        {
            lastInput = input;
            return input.ApplyFunctionOnAllElements(x => x > 0 ? x : 0);
        }

        public Tensor BackwardPass(Tensor previousGradient)
        {
            Func<double, double, double> reluDer = (double data, double previous) =>
                previous > 0 ? data : 0;
            return previousGradient.ApplyFunctionOnAllElements(reluDer, lastInput);
        }
    }
}
