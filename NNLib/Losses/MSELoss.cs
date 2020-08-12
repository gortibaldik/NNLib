using System;
namespace NNLib.Losses
{
    public class MSELoss : ILossLayer
    {
        Tensor neuralOutput = null;
        Tensor expectedOutput = null;

        public Tensor ForwardPass(Tensor neuralOutput, Tensor expectedOutput)
        {
            if (neuralOutput.Rows != expectedOutput.Rows || neuralOutput.Columns != expectedOutput.Columns || neuralOutput.Depth != expectedOutput.Depth)
                throw new InvalidOperationException("The output of the neural net doesn't correspond to the expected output!");

            this.neuralOutput = neuralOutput;
            this.expectedOutput = expectedOutput;
            Tensor result = new Tensor(1,1, 1);
            var res = 0D;

            for (int d = 0; d < neuralOutput.Depth; d++)
                for (int r = 0; r < neuralOutput.Rows; r++)
                    for (int c = 0; c < neuralOutput.Columns; c++)
                        res += Math.Pow(neuralOutput[d,r, c] - expectedOutput[d,r, c], 2);

            result[0, 0, 0] = res;
            return result;
        }

        public Tensor BackwardPass()
        {
            if (neuralOutput == null || expectedOutput == null)
                throw new InvalidOperationException("Backward pass before forward pass exception !");

            Tensor result = new Tensor(neuralOutput.Depth, neuralOutput.Rows, neuralOutput.Columns);

            for (int d = 0; d < neuralOutput.Depth; d++)
                for (int r = 0; r < neuralOutput.Rows; r++)
                    for (int c = 0; c < neuralOutput.Columns; c++)
                        result[d, r, c] = 2*(neuralOutput[d, r, c] - expectedOutput[d, r, c]);

            neuralOutput = null;
            expectedOutput = null;
            return result;
        }
    }
}
