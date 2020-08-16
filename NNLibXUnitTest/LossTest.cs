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
        public void SparseCategoricalCrossEntropyForward()
        {

        }
    }
}
