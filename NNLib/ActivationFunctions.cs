using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib
{
    public class LinearActivation : ActivationLayer
    {
        public Tensor ForwardPass(Tensor input)
            => input;

        public Tensor BackwardPass(Tensor previousGradient)
            => previousGradient;
    }

    public class ReLUActivation : ActivationLayer
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
