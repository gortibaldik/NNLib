namespace NNLib.Activations
{
    public class LinearActivation : IActivationLayer
    {
        public ActivationFunctions Name { get => ActivationFunctions.Linear; }
        public Tensor ForwardPass(Tensor input)
            => input;

        public Tensor BackwardPass(Tensor previousGradient)
            => previousGradient;
    }
}
