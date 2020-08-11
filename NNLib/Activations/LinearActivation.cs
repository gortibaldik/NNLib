namespace NNLib.Activations
{
    public class LinearActivation : IActivationLayer
    {
        public Tensor ForwardPass(Tensor input)
            => input;

        public Tensor BackwardPass(Tensor previousGradient)
            => previousGradient;
    }
}
