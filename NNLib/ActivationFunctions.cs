using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib
{
    public class LinearActivation : ActivationLayer
    {
        public Matrix ForwardPass(Matrix input)
            => input;

        public Matrix BackwardPass(Matrix previousGradient)
            => previousGradient;
    }

    public class ReLUActivation : ActivationLayer
    {
        private Matrix lastInput;
        public Matrix ForwardPass(Matrix input)
        {
            lastInput = input;
            return input.ApplyFunctionOnAllElements(x => x > 0 ? x : 0);
        }

        public Matrix BackwardPass(Matrix previousGradient)
        {
            Func<double, double, double> reluDer = (double data, double previous) =>
                previous > 0 ? data : 0;
            return previousGradient.ApplyFunctionOnAllElements(reluDer, lastInput);
        }
    }
}
