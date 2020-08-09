using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib
{
    public class MSELoss : LossLayer
    {
        Matrix neuralOutput = null;
        Matrix expectedOutput = null;

        public Matrix ForwardPass(Matrix neuralOutput, Matrix expectedOutput)
        {
            if (neuralOutput.Rows != expectedOutput.Rows || neuralOutput.Columns != expectedOutput.Columns)
                throw new InvalidOperationException("The output of the neural net doesn't correspond to the expected output!");

            this.neuralOutput = neuralOutput;
            this.expectedOutput = expectedOutput;
            Matrix result = new Matrix(1, 1);

            for (int r = 0; r < neuralOutput.Rows; r++)
                for (int c = 0; c < neuralOutput.Columns; c++)
                    result[0,0] += Math.Pow(neuralOutput[r, c] - expectedOutput[r, c], 2);

            return result;
        }

        public Matrix BackwardPass()
        {
            if (neuralOutput == null || expectedOutput == null)
                throw new InvalidOperationException("Backward pass before forward pass exception !");

            Matrix result = new Matrix(neuralOutput.Rows, neuralOutput.Columns);

            for (int r = 0; r < neuralOutput.Rows; r++)
                for (int c = 0; c < neuralOutput.Columns; c++)
                    result[r, c] = 2*(neuralOutput[r, c] - expectedOutput[r, c]);

            neuralOutput = null;
            expectedOutput = null;
            return result;
        }
    }
}
