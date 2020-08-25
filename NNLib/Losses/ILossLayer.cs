namespace NNLib.Losses
{ 
    public interface ILossLayer
    {
        /// <summary>
        /// Calculates the loss with respect to all the batches of the 
        /// neural network. Returns double, the average loss across
        /// all the batches.
        /// </summary>
        double ForwardPass(Tensor neuralOutput, Tensor expectedOutput);

        /// <summary>
        /// Calculates the gradient of the loss with respect to the
        /// output of the neural network. 
        /// ForwardPass must be called before the call to BackwardPass,
        /// otherwise InvalidOperationException is thrown.
        /// </summary>
        Tensor BackwardPass();
    }
}
