using System;

namespace NNLib.Activations
{
    public class ReLU : IActivationLayer
    {
        private Tensor lastInput;
        private bool forwardPerformed = false;

        public Tensor ForwardPass(Tensor input)
        {
            lastInput = input;
            forwardPerformed = true;
            return input.ApplyFunctionOnAllElements(x => x > 0 ? x : 0);
        }

        public Tensor BackwardPass(Tensor previousGradient)
        {
            if (!forwardPerformed)
                throw new InvalidOperationException("No forward pass before backward pass !");

            Func<double, double, double> reluDer = (double data, double previous) =>
                previous > 0 ? data : 0;
            
            forwardPerformed = false;
            return previousGradient.ApplyFunctionOnAllElements(reluDer, lastInput);
        }
    }
}
