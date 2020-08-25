using System;

namespace NNLib.Activations
{
    public class LinearActivation : IActivationLayer
    {
        private bool forwardPerformed = false;

        public Tensor ForwardPass(Tensor input)
        {
            forwardPerformed = true;
            return input;
        }

        public Tensor BackwardPass(Tensor previousGradient)
        {
            if (!forwardPerformed)
                throw new InvalidOperationException("No forward pass before backward pass !");

            forwardPerformed = false;
            return previousGradient;
        }
    }
}
