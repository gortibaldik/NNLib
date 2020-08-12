namespace NNLib.Losses
{ 
    public interface ILossLayer
    {
        double ForwardPass(Tensor neuralOutput, Tensor expectedOutput);

        Tensor BackwardPass();
    }
}
