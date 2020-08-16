using System;
using System.Runtime.CompilerServices;
using Xunit;
using NNLib;
using Xunit.Sdk;
using Microsoft.VisualStudio.TestPlatform.ObjectModel;

namespace NNLibXUnitTest
{
    public class TensorTest
    {
        [Fact]
        public void ParallelCopyTo1()
        {
            // arrange
            int capacity = 20_003;
            var result = new double[capacity];
            for (int i = 0; i < capacity; i++)
                result[i] = i * 5;

            var d = new double[capacity];

            // act
            result.ParallelCopyTo(d);

            // assert
            for (int i = 0; i < 200; i++)
                Assert.Equal(result[i], d[i]);
        }

        [Fact]
        public void ParallelCopyTo2()
        {
            // arrange
            int dims = 100;
            int capacity = 5000;
            var result = new double[dims][];
            for (int j = 0; j < dims; j++)
            {
                result[j] = new double[capacity];
                for (int i = 0; i < capacity; i++)
                    result[j][i] = i;
            }

            var d = new double[capacity * dims];

            // act
            result.ParallelCopyTo(d);

            // assert
            for (int j = 0; j < dims; j++)
            {
                var offset = j * capacity;
                for (int i = 0; i < capacity; i++, offset++)
                    Assert.Equal(result[j][i], d[offset]);
            }
        }

        [Fact]
        public void Reshape1()
        {
            //arrange
            var t1 = new Tensor(new double[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, { { { 9, 10 }, { 11, 12 } }, { { 13, 14 }, { 15, 16 } } } });
            var r1 = new double[,,,] { { { { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 } } } };

            //act
            var t2 = t1.Reshape(1, 1, 1, 16);

            // assert
            for (int b = 0; b < r1.GetLength(0); b++)
                for (int d = 0; d < r1.GetLength(1); d++)
                    for (int r = 0; r < r1.GetLength(2); r++)
                        for (int c = 0; c < r1.GetLength(3); c++)
                            Assert.Equal(r1[b, d, r, c], t2[b, d, r, c]);
        }

        [Fact]
        public void Reshape2()
        {
            //arrange
            var t1 = new Tensor(new double[,,,] { { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, { { { 9, 10 }, { 11, 12 } }, { { 13, 14 }, { 15, 16 } } } });
            var r2 = new double[,,,] { { { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }, { 10 }, { 11 }, { 12 }, { 13 }, { 14 }, { 15 }, { 16 } } } };

            //act
            var t3 = t1.Reshape(1, 1, 16, 1);

            // assert
            for (int b = 0; b < r2.GetLength(0); b++)
                for (int d = 0; d < r2.GetLength(1); d++)
                    for (int r = 0; r < r2.GetLength(2); r++)
                        for (int c = 0; c < r2.GetLength(3); c++)
                            Assert.Equal(r2[b, d, r, c], t3[b, d, r, c]);
        }

        [Fact]
        public void SimpleMatrixMultiplication1()
        {
            // arrange
            var t1 = new Tensor(new double[,,,] { { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } } });
            var t2 = new Tensor(new double[,,,] { { { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } } } });
            var r = new double[,,,] { { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } } };

            // act
            var t3 = t1 * t2;

            // assert
            for (int b = 0; b < r.GetLength(0); b++)
                for (int d = 0; d < r.GetLength(1); d++)
                    for (int row = 0; row < r.GetLength(2); row++)
                        for (int c = 0; c < r.GetLength(3); c++)
                            Assert.Equal(r[b, d, row, c], t3[b, d, row, c]);
        }

        [Fact]
        public void TensorMultiplicationSameDepthSameBatchSize1()
        {
            // arrange
            var t1 = new Tensor(new double[,,,] { { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } }, { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } } });
            var t2 = new Tensor(new double[,,,] { { { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }, { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } } } });
            var r = new double[,,,] { { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } }, { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } } };

            // act
            var t3 = t1 * t2;

            // assert
            for (int b = 0; b < r.GetLength(0); b++)
                for (int d = 0; d < r.GetLength(1); d++)
                    for (int row = 0; row < r.GetLength(2); row++)
                        for (int c = 0; c < r.GetLength(3); c++)
                            Assert.Equal(r[b, d, row, c], t3[b, d, row, c]);
        }

        [Fact]
        public void TensorMultiplicationSameDepthSameBatchSize2()
        {
            // arrange
            var t1 = new Tensor(new double[,,,] { { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } }, { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } }, { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } }, { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } } });
            var t2 = new Tensor(new double[,,,] { { { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }, { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } } }, { { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }, { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } } } });
            var r = new double[,,,] { { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } }, { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } }, { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } }, { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } } } };

            // act
            var t3 = t1 * t2;

            // assert
            for (int b = 0; b < r.GetLength(0); b++)
                for (int d = 0; d < r.GetLength(1); d++)
                    for (int row = 0; row < r.GetLength(2); row++)
                        for (int c = 0; c < r.GetLength(3); c++)
                            Assert.Equal(r[b, d, row, c], t3[b, d, row, c]);
        }

        [Fact]
        public void TensorMultiplicationDifferentDepthSameBatchSize1()
        {
            // arrange
            var t1 = new Tensor(new double[,,,] { { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } }, { { 11, 12, 13 }, { 14, 15, 16 }, { 17, 18, 19 } } }, { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } }, { { 11, 12, 13 }, { 14, 15, 16 }, { 17, 18, 19 } } } });
            var t2 = new Tensor(new double[,,,] { { { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } } }, { { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } } } });
            t2.Mode = TensorMultiplicationModes.LastLevel;
            var r = new double[,,,] { { { { 11, 12, 13 }, { 14, 15, 16 }, { 17, 18, 19 } } }, { { { 11, 12, 13 }, { 14, 15, 16 }, { 17, 18, 19 } } } };

            // act
            var t3 = t2 * t1;

            // assert
            for (int b = 0; b < r.GetLength(0); b++)
                for (int d = 0; d < r.GetLength(1); d++)
                    for (int row = 0; row < r.GetLength(2); row++)
                        for (int c = 0; c < r.GetLength(3); c++)
                            Assert.Equal(r[b, d, row, c], t3[b, d, row, c]);
        }

        [Fact]
        public void TensorMultiplicationSameDepthDifferentBatchSize()
        {
            // arrange
            var t1 = new Tensor(new double[,,,] { { { { 1, 1, 1 } } } });
            var t2 = new Tensor(new double[,,,] { { { { 1, 2 }, { 1, 3 }, { 1, 4 } } }, { { { 2, 5 }, { 2, 6 }, { 2, 7 } } } });
            var r = new double[,,,] { { { { 3, 9 } } }, { { { 6, 18 } } } };

            // act
            var t3 = t1 * t2;
            // assert
            for (int b = 0; b < r.GetLength(0); b++)
                for (int d = 0; d < r.GetLength(1); d++)
                    for (int row = 0; row < r.GetLength(2); row++)
                        for (int c = 0; c < r.GetLength(3); c++)
                            Assert.Equal(r[b, d, row, c], t3[b, d, row, c]);

        }

        [Fact]
        public void TensorAdditionSameDimensions1()
        {
            // arrange
            var t1 = new Tensor(new double[,,,] { { { { 8, 7, 6, 5 }, { 4, 3, 2, 1 } } } });
            var t2 = new Tensor(new double[,,,] { { { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } } } });

            // act
            var t3 = t1 + t2;

            // assert
            for (int r = 0; r < 2; r++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(9, t3[0, 0, r, c]);
        }

        [Fact]
        public void ElementWiseAdditionDifferentColumnsSameBatches()
        {
            // arrange
            var t1 = new Tensor(new double[,,,] { { { { 1, 2, 3, 4 }, { 3, 4, 5, 6 } } } });
            var t2 = new Tensor(new double[,,,] { { { { 1 }, { 3 } } } });
            var result = new double[,,,] { { { { 2, 3, 4, 5 }, { 6, 7, 8, 9 } } } };

            // act
            var t3 = t1 + t2;

            // assert
            for (int r = 0; r < 2; r++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(result[0, 0, r, c], t3[0, 0, r, c]);
        }

        [Fact]
        public void ElementWiseAdditionSameColumnsDifferentBatches()
        {
            // arrange
            var t1 = new Tensor(new double[,,,] { { { { 8, 7, 6, 5 }, { 4, 3, 2, 1 } } }, { { { -1, -2, -3, -4}, { -5, -6, -7, -8 } } } });
            var t2 = new Tensor(new double[,,,] { { { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } } } });
            var result = new double[,,,] { { { { 9, 9, 9, 9 }, { 9, 9, 9, 9 } } }, { { { 0, 0, 0, 0 }, { 0, 0, 0, 0 } } } };

            // act
            var t3 = t1 + t2;

            // assert
            for (int b = 0; b < 2; b++)
                for (int r = 0; r < 2; r++)
                    for (int c = 0; c < 3; c++)
                        Assert.Equal(result[b, 0, r, c], t3[b, 0, r, c]);
        }

        [Fact]
        public void ElementWiseAdditionDifferentColumnsDifferentBatches()
        {
            // arrange
            var t1 = new Tensor(new double[,,,] { { { { 1, 2, 3, 4 }, { 3, 4, 5, 6 } } }, { { { 2, 3, 4, 5 }, { 4, 5, 6, 7 } } } });
            var t2 = new Tensor(new double[,,,] { { { { 1 }, { 3 } } } });
            var result = new double[,,,] { { { { 2, 3, 4, 5 }, { 6, 7, 8, 9 } } }, { { { 3, 4, 5, 6 }, { 7, 8, 9, 10 } } } };

            // act
            var t3 = t1 + t2;

            // assert
            for (int r = 0; r < 2; r++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(result[0, 0, r, c], t3[0, 0, r, c]);
        }

        [Fact]
        public void SumRowsTest1()
        {
            // arrange
            var t1 = new Tensor(new double[,,,] { { { { 1, 2, 3 }, { 4, 5, 6 } } }, { { { 1, 2, 3 }, { 4, 5, 6 } } } });
            var r =new double[,,,] { { { { 6 }, { 15 } } }, { { { 6 }, { 15 } } } };

            // act
            var t2 = t1.SumRows();

            // assert
            for (int b = 0; b < r.GetLength(0); b++)
                for (int d = 0; d < r.GetLength(1); d++)
                    for (int row = 0; row < r.GetLength(2); row++)
                        for (int c = 0; c < r.GetLength(3); c++)
                            Assert.Equal(r[b, d, row, c], t2[b, d, row, c]);
        }

        [Fact]
        public void SumRowsTest2()
        {
            // arrange
            var t1 = new Tensor(new double[,,,] { { { { 1, 2, 3 }, { 4, 5, 6 } }, { { 1, 2, 3 }, { 4, 5, 6 } } } });
            var r = new double[,,,] { { { { 6 }, { 15 } }, { { 6 }, { 15 } } } };

            // act
            var t2 = t1.SumRows();

            // assert
            for (int b = 0; b < r.GetLength(0); b++)
                for (int d = 0; d < r.GetLength(1); d++)
                    for (int row = 0; row < r.GetLength(2); row++)
                        for (int c = 0; c < r.GetLength(3); c++)
                            Assert.Equal(r[b, d, row, c], t2[b, d, row, c]);
        }

        [Fact]
        public void SumBatchTest1()
        {
            // arrange
            int BatchSize = 3, Rows = 28, Columns = 28;
            var t1 = new Tensor(BatchSize, 1, Rows, Columns);
            for (int b = 0; b < BatchSize; b++)
                for (int r = 0; r < Rows; r++)
                    for (int c = 0; c < Columns; c++)
                        t1[b, 0, r, c] = b*(r + c);

            // act
            var t2 = t1.SumBatch();

            // assert
            for (int r = 0; r < Rows; r++)
                for (int c = 0; c < Columns; c++)
                    Assert.Equal(3*(r+c), t2[0, 0, r, c]);
        }
    }
}
