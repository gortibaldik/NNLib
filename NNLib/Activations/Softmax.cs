using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace NNLib.Activations
{
    /// <summary>
    /// Only one supported mode, with 1xNx1 tensor shape at the input and softmax right before CrossEntropy Loss
    /// </summary>
    public class Softmax : IActivationLayer
    {
        public ActivationFunctions Name { get => ActivationFunctions.Softmax; }

        /// <summary>
        /// This isn't real and shouldn't be considered as a real implementation of the softmax activation. 
        /// Neural network itself will check if 
        /// </summary>
        /// <param name="previousGradient"></param>
        /// <returns></returns>
        public Tensor BackwardPass(Tensor previousGradient)
            => previousGradient;

        public Tensor BackwardPassFromCrossEntropy(Tensor previousGradient, bool alreadyComputed = true)
            => alreadyComputed ? previousGradient : BackwardPass(previousGradient);

        public Tensor ForwardPass(Tensor input)
        {
            // not numerically stable!
            if (input.Depth != 1 || input.Columns != 1)
                throw new NotImplementedException("The softmax layer doesn't support other than 1 x N x 1 shape of the tensor!");

            var softmax = new Tensor(input.BatchSize, 1, input.Rows, 1);
            var exponentiated = new Tensor(1, 1, input.Rows, 1);
            for (int b = 0; b < input.BatchSize; b++)
            {
                var max = double.MinValue;
                var sum = 0D;
                for (int r = 0; r < input.Rows; r++)
                {
                    if (input[b, 0, r, 0] > max)
                        max = input[b, 0, r, 0];
                }
                
                for (int r = 0; r < input.Rows; r++)
                {
                    var d = Math.Exp(input[b, 0, r, 0] - max);
                    exponentiated[0, 0, r, 0] = d;
                    sum += d;
                }

                for (int r = 0; r < input.Rows; r++)
                {
                    softmax[b, 0, r, 0] = exponentiated[0, 0, r, 0] / sum;
                }
            }
            //var exponentiated = input.ApplyFunctionOnAllElements(x => { var d = Math.Exp(x-max); sum += d; return d; });
            //var softmax = exponentiated.ApplyFunctionOnAllElements(x => x / sum);

            return softmax;
        }
    }
}
