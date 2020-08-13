namespace NNLib.Activations
{
    public interface IActivationLayer
    {
        ActivationFunctions Name { get; }
        Tensor ForwardPass(Tensor input);

        Tensor BackwardPass(Tensor previousGradient);
    }

    public enum ActivationFunctions
    {
        ReLU,
        Softmax, 
        Linear
    }
}
