using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace NNLib.Activations
{
    /// <summary>
    /// Implementation cannot be considered as complete or even partially complete. 
    /// Only one supported mode, with SparseCategoricalCrossEntropy directly following
    /// the softmax and softmax.ForwardPass accepting only tensors of shape B x 1 x R x 1
    /// </summary>
    public class Softmax : IActivationLayer
    {

        /// <summary>
        /// Only one supported mode, with SparseCategoricalCrossEntropy directly following
        /// the softmax. Hence all the computation of gradient of SparseCategoricalCrossentropy(Softmax(neuralOutput)) 
        /// with respect to neuralOutput is left for SparseCategoricalCrossentropy.BackwardPass
        /// </summary>
        /// <returns>Exactly the input.</returns>
        public Tensor BackwardPass(Tensor previousGradient)
            => previousGradient;

        public Tensor ForwardPass(Tensor input)
        {
            if (input.Depth != 1 || input.Columns != 1)
                throw new NotImplementedException("The softmax layer doesn't support other than B x 1 x N x 1 shape of the tensor !");

            var softmax = new Tensor(input.BatchSize, 1, input.Rows, 1);
            var exponentiated = new Tensor(1, 1, input.Rows, 1);

            for (int b = 0; b < input.BatchSize; b++)
            {
                var max = double.MinValue;
                var sum = 0D;

                // For the numerical stability normalization of 
                // each row is performed.
                // Since exp() is monotonic
                // if exp(x) < exp(y) then exp(x-max) < exp(y-max)

                // finding maximum of each batch
                for (int r = 0; r < input.Rows; r++)
                {
                    if (input[b, 0, r, 0] > max)
                        max = input[b, 0, r, 0];
                }
                
                // normalizing each input and exponentiating
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

            return softmax;
        }
    }
}
