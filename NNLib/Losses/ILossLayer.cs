namespace NNLib.Losses
{ 
    public interface ILossLayer
    {
        Tensor ForwardPass(Tensor neuralOutput, Tensor expectedOutput);

        Tensor BackwardPass();
    }
}
