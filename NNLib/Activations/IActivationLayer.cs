namespace NNLib.Activations
{
    public interface IActivationLayer
    {
        /// <summary>
        /// Outputs the tensor created by applying the activation on the input.
        /// May cache the input for the BackwardPass
        /// </summary>
        Tensor ForwardPass(Tensor input);

        /// <summary>
        /// Performs gradient computation and returns the output. Should check if the 
        /// forward pass was called before backward pass and throw InvalidOperationException
        /// if not.
        /// </summary>
        Tensor BackwardPass(Tensor previousGradient);
    }
}
