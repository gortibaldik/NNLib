using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib.Losses
{
    public class SparseCategoricalCrossEntropy : ILossLayer
    {
        private Tensor neuralOutput;
        private Tensor expectedOutput;

        public Tensor BackwardPass()
        {
            var result = new Tensor(expectedOutput.Depth, expectedOutput.Rows, expectedOutput.Columns);

            for (int d = 0; d < expectedOutput.Depth; d++)
                for (int r = 0; r < expectedOutput.Rows; r++)
                    result[d, r, 0] = neuralOutput[d, r, 0] - expectedOutput[d, r, 0];

            return result;
        }

        public double ForwardPass(Tensor neuralOutput, Tensor expectedOutput)
        {
            if (neuralOutput.Rows != expectedOutput.Rows || neuralOutput.Columns != expectedOutput.Columns || neuralOutput.Depth != expectedOutput.Depth)
                throw new InvalidOperationException("The output of the neural net doesn't correspond to the expected output!");

            this.neuralOutput = neuralOutput;
            this.expectedOutput = expectedOutput;
            var res = 0D;

            for (int d = 0; d < neuralOutput.Depth; d++)
                for (int r = 0; r < neuralOutput.Rows; r++)
                    res -= expectedOutput[d,r,0]*Math.Log(neuralOutput[d,r,0]);

            return res;
        }
    }
}
