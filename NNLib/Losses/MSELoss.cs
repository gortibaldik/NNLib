using System;
namespace NNLib.Losses
{
    public class MSELoss : ILossLayer
    {
        Tensor neuralOutput = null;
        Tensor expectedOutput = null;

        public double ForwardPass(Tensor neuralOutput, Tensor expectedOutput)
        {
            if (neuralOutput.Rows != expectedOutput.Rows || neuralOutput.Columns != expectedOutput.Columns || neuralOutput.Depth != expectedOutput.Depth
                || neuralOutput.BatchSize != expectedOutput.BatchSize)
                throw new InvalidOperationException("The output of the neural net doesn't correspond to the expected output!");

            this.neuralOutput = neuralOutput;
            this.expectedOutput = expectedOutput;
            var res = 0D;

            // since neuralOutput.ApplyFunctionOnAllElements is performed without parallelism
            // we can call the function as an aggregate with var res collecting
            // the result
            neuralOutput.ApplyFunctionOnAllElements((got, expected) =>
                {
                    res += Math.Pow(got - expected, 2);
                    return got;
                }, expectedOutput, disableChecking : true);

            return res/ neuralOutput.BatchSize;
        }

        public Tensor BackwardPass()
        {
            if (neuralOutput == null || expectedOutput == null)
                throw new InvalidOperationException("Backward pass before forward pass exception !");

            var result = neuralOutput.ApplyFunctionOnAllElements((got, expected) => 2 * (got - expected), expectedOutput, disableChecking : true);

            neuralOutput = null;
            expectedOutput = null;
            return result;
        }
    }
}
