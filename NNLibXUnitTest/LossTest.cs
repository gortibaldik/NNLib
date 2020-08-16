using System;
using System.Runtime.CompilerServices;
using Xunit;
using NNLib;
using NNLib.Losses;
using Xunit.Sdk;

namespace NNLibXUnitTest
{
    public class LossTest
    {
        [Fact]
        public void MSELossForward()
        {
            // arrange
            var t1 = new Tensor(new double[,,,] { { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } } });
            var t2 = new Tensor(new double[,,,] { { { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } } } });
            var expected = 258D;
            var loss = new MSELoss();

            // act
            var actual = loss.ForwardPass(t1, t2);

            // assert
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void MSELossBackward()
        {
            // arrange
            var t1 = new Tensor(new double[,,,] { { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } } });
            var t2 = new Tensor(new double[,,,] { { { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } } } });
            var expected = new double[,,,] { { { { 0, 4, 6 }, { 8, 8, 12 }, { 14, 16, 16 } } } };
            var loss = new MSELoss();
            loss.ForwardPass(t1, t2);

            // act
            var actual = loss.BackwardPass();

            // assert
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(expected[0, 0, r, c], actual[0, 0, r, c]);
        }

        [Fact]
        public void SparseCategoricalCrossEntropyForward1()
        {
            // arrange
            var probs = new Tensor(new double[,,,] { { { { 5 }, { 2 }, { 3 }, { 1 } } } });
            var truee = new Tensor(new double[,,,] { { { { 7 }, { 2 }, { 3 }, { 3 } } } });
            var expected = -(7 * Math.Log(5) + 2 * Math.Log(2) + 3 * Math.Log(3) + 3 * Math.Log(1));
            var loss = new SparseCategoricalCrossEntropy();

            // act
            var actual = loss.ForwardPass(probs, truee);

            // assert
            Assert.Equal(expected, actual);
        }

        [Fact]
        public void SparseCategoricalCrossEntropyForward2()
        {
            // arrange
            var probs = new Tensor(new double[,,,] { { { { 5 }, { 2 }, { 3 }, { 1 } } }, { { { 4 }, { 8 }, { 2 }, { 5 } } } });
            var truee = new Tensor(new double[,,,] { { { { 7 }, { 2 }, { 3 }, { 3 } } }, { { { 5 }, { 9 }, { 12 }, { 13 } } } });
            var expected = -(7 * Math.Log(5) + 2 * Math.Log(2) + 3 * Math.Log(3) + 3 * Math.Log(1)) - (5 * Math.Log(4) + 9 * Math.Log(8) + 12 * Math.Log(2) + 13 * Math.Log(5));
            expected /= 2;
            var loss = new SparseCategoricalCrossEntropy();

            // act
            var actual = loss.ForwardPass(probs, truee);

            // assert
            Assert.Equal(2, probs.BatchSize);
            Assert.True(probs.BatchSize == truee.BatchSize);
            Assert.Equal(expected, actual);
        }
    }
}
