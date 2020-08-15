using System;
using System.Runtime.CompilerServices;
using Xunit;
using NNLib;
using Xunit.Sdk;

namespace NNLibXUnitTest
{
    public class TensorTest
    {
        [Fact]
        public void ParallelCopyTo()
        {
            // arrange
            int capacity = 20_003;
            var result = new double[capacity];
            for (int i = 0; i < capacity; i++)
                result[i] = i*5;

            var d = new double[capacity];

            // act
            result.ParallelCopyTo(d);

            // assert
            for (int i = 0; i < 200; i++)
                Assert.Equal(result[i], d[i]);
        }


        [Fact]
        public void ElementWiseAdditionSameDimensions1()
        {
            // arrange
            var t1 = new Tensor(new double[,,] { { { 0, 0, 0, 0 }, { 0, 0, 0, 0 } } });
            var result = new double[,,] { { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } } };
            var t2 = new Tensor(result);

            // act
            var t3 = t1 + t2;

            // assert
            for (int r = 0; r < 2; r++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(result[0, r, c], t3[0, r, c]);
        }

        [Fact]
        public void ElementWiseAdditionSameDimensions2()
        {
            // arrange
            var result = new double[,,] { { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } } };
            var t1 = new Tensor(result);
            var t2 = new Tensor(new double[,,] { { { 0, 0, 0, 0 }, { 0, 0, 0, 0 } } });

            // act
            var t3 = t1 + t2;

            // assert
            for (int r = 0; r < 2; r++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(result[0, r, c], t3[0, r, c]);
        }

        [Fact]
        public void ElementWiseAdditionNotSameDimensions1()
        {
            // arrange
            var t1 = new Tensor(new double[,,] { { { 0, 0, 0, 0 }, { 0, 0, 0, 0 } } });
            var t2 = new Tensor(new double[,,] { { { 1 }, { 2 } } });
            var result = new double[,,] { { { 1, 1, 1, 1 }, { 2, 2, 2, 2 } } };

            // act
            var t3 = t1 + t2;

            // assert
            for (int r = 0; r < 2; r++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(result[0, r, c], t3[0, r, c]);
        }

        [Fact]
        public void TensorMultiplicationTest1()
        {
            // arrange
            var t1 = new Tensor(new double[,,] { { { 1, 2, 3 } } });
            var t2 = new Tensor(new double[,,] { { { 4 }, { 5 }, { 6 } } });
            var result = new double[,,] { { { 32 } } };

            // act
            var t3 = t1 * t2;

            // assert
            Assert.Equal(result[0, 0, 0], t3[0, 0, 0]);
            Assert.Equal(1, t3.Depth);
            Assert.Equal(1, t3.Rows);
            Assert.Equal(1, t3.Columns);
        }

        [Fact]
        public void TensorMultiplicationTest2()
        {
            // arrange
            var t1 = new Tensor(new double[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } });
            var t2 = new Tensor(new double[,,] { { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } } });
            var result = new double[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } };

            // act
            var t3 = t1 * t2;

            // assert
            Assert.Equal(1, t3.Depth);
            Assert.Equal(2, t3.Rows);
            Assert.Equal(3, t3.Columns);
            for (int r = 0; r < t3.Rows; r++)
                for (int c = 0; c < t3.Columns; c++)
                    Assert.Equal(result[0, r, c], t3[0, r, c]);
        }

        [Fact]
        public void SumRowsTest1()
        {
            // arrange
            var t1 = new Tensor(new double[,,] { { { 1, 2, 3 }, { 4, 5, 6 } } });
            var result = new Tensor(new double[,,] { { { 6 }, { 15 } } });

            // act
            var t2 = t1.SumRows();

            // assert
            for (int r = 0; r < t2.Rows; r++)
                for (int c = 0; c < t2.Columns; c++)
                    Assert.Equal(result[0, r, c], t2[0, r, c]);
        }

        [Fact]
        public void SumRowsTest2()
        {
            // arrange
            var t1 = new Tensor(new double[,,] { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 1, 2, 3 }, { 4, 5, 6 } } });
            var result = new Tensor(new double[,,] { { { 6 }, { 15 } }, { { 6 }, { 15 } } });

            // act
            var t2 = t1.SumRows();

            // assert
            for (int d = 0; d < 2; d++)
                for (int r = 0; r < t2.Rows; r++)
                    for (int c = 0; c < t2.Columns; c++)
                        Assert.Equal(result[d, r, c], t2[d, r, c]);
        }
    }
}
