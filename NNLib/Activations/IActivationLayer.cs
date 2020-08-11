namespace NNLib.Activations
{
    public interface IActivationLayer
    {
        Tensor ForwardPass(Tensor input);

        Tensor BackwardPass(Tensor previousGradient);
    }
}
