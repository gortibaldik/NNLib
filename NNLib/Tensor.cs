#define FORCE_PARALLELISM

using System;
using System.Globalization;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Schema;

namespace NNLib
{
    public enum TensorMultiplicationModes
    {
        OnlySameDepth,
        LastLevel
    }

    public class Tensor
    {
        private double[] data;

        public int BatchSize { get; }
        public int Depth { get; }
        public int Rows { get; }
        public int Columns { get; }
        public TensorMultiplicationModes Mode { get; set; } = TensorMultiplicationModes.OnlySameDepth;


        internal int RowsColumns { get; }
        internal int DepthRowsColumns { get; }

        public Tensor(int depth, int rows, int columns, double[][] data)
        {
            if (data[0].Length != depth * rows * columns)
                throw new ArgumentException($"array {nameof(data)} doesn't contain enough elements to fill {depth}x{rows}x{columns} tensor !");

            Depth = depth;
            Rows = rows;
            Columns = columns;
            BatchSize = data.Length;
            RowsColumns = rows * columns;
            DepthRowsColumns = Depth * RowsColumns;
            this.data = new double[data.Length*data[0].Length];
            data.ParallelCopyTo(this.data);
        }

        public Tensor(double[,,,] data)
        {
            BatchSize = data.GetLength(0);
            Depth = data.GetLength(1);
            Rows = data.GetLength(2);
            Columns = data.GetLength(3);

            RowsColumns = Rows * Columns;
            DepthRowsColumns = Depth * RowsColumns;
            this.data = new double[BatchSize * Depth * Rows * Columns];

            for (int b = 0; b < BatchSize; b++)
                for (int d = 0; d < Depth; d++)
                    for (int r = 0; r < Rows; r++)
                        for (int c = 0; c < Columns; c++)
                            this[b,d,r,c] = data[b,d,r,c];
        }

        public Tensor(int batchSize, int depth, int rows, int columns)
        {
            Rows = rows;
            Columns = columns;
            Depth = depth;
            BatchSize = batchSize;
            RowsColumns = rows * columns;
            DepthRowsColumns = Depth * RowsColumns;
            data = new double[batchSize*depth*rows*columns];
        }

        public Tensor(int batchSize, int depth, int rows, int columns, NInitializer nInit) : this(batchSize, depth, rows, columns)
        { 
            for (int i = 0; i < data.Length; i++)
                this.data[i] = nInit();
        }

        public double this[int batchElement, int depth, int row, int column]
        {
            get
            {
                if (batchElement < BatchSize && depth < Depth && row < Rows && column < Columns)
                    return data[batchElement * DepthRowsColumns + depth * RowsColumns + row * Columns + column];

                throw new IndexOutOfRangeException();
            }
            internal set => data[batchElement*DepthRowsColumns + depth*RowsColumns + row*Columns + column] = value;
        }

        public Tensor Reshape(int newBatchSize, int newDepth, int newRows, int newColumns)
        {
            if (Depth * Rows * Columns * BatchSize != newRows * newColumns * newDepth * newBatchSize)
                throw new ArgumentOutOfRangeException($"Cannot reshape {BatchSize}*{Depth}*{Rows}*{Columns} to {newBatchSize}*{newDepth}*{newRows}*{newColumns} !");

            if (newRows < 0 || newColumns < 0 || newDepth < 0 || newBatchSize < 0)
                throw new ArgumentOutOfRangeException("Dimensions of matrix cannot be less than or equal to zero !");

            var result = new Tensor(newBatchSize, newDepth, newRows, newColumns);

            data.ParallelCopyTo(result.data);

            return result;
        }

        public Tensor Transpose()
        {
            Tensor result = new Tensor(BatchSize, Depth, Columns, Rows);
            for (int b = 0; b < BatchSize; b++)
                for (int d = 0; d < Depth; d++)
                    Parallel.For(0, Rows, r =>
                    {
                        for (int c = 0; c < Columns; c++)
                            result[b, d, c, r] = this[b, d, r, c];
                    });

            return result;
        }

        public static Tensor operator *(Tensor t1, Tensor t2)
        {
            if (t1.Columns != t2.Rows)
                throw new ArgumentException($"TENSOR MULTIPLICATION : {nameof(t1.Columns)} not equal to {nameof(t2.Rows)} !");

            if (t1.BatchSize == t2.BatchSize)
                return _sameBatchSizeMultiplication(t1, t2);

            else
                return _differentBatchSizeMultiplication(t1, t2);
        }

        private static Tensor _sameBatchSizeMultiplication(Tensor t1, Tensor t2)
        {
            if (t1.Depth == t2.Depth)
            {
                var t1Offset = 0;
                var t2Offset = 0;
                var result = new Tensor(t2.BatchSize, t1.Depth, t1.Rows, t2.Columns);

                for (int b = 0; b < t2.BatchSize; b++)
                    for (int d = 0; d < t1.Depth; d++, t1Offset += t1.RowsColumns, t2Offset += t2.RowsColumns)
                        _simpleMatrixMultiplication(b, d, t1Offset, t2Offset, t1, t2, result);

                return result;
            }
            else if (t1.Depth == 1 && t1.Mode == TensorMultiplicationModes.LastLevel)
            {
                var t2Offset = (t2.Depth - 1) * t2.RowsColumns;
                var t1Offset = 0;
                var result = new Tensor(t1.BatchSize, 1, t1.Rows, t2.Columns);


                for (int b = 0; b < t2.BatchSize; b++, t1Offset += t1.DepthRowsColumns, t2Offset += t2.DepthRowsColumns)
                    _simpleMatrixMultiplication(b, 0, t1Offset, t2Offset, t1, t2, result);

                return result;
            }

            else
                throw new InvalidOperationException($"TENSOR MULTIPLICATION : non supported values of {nameof(t1.Depth)} : {t1.Depth}, {t2.Depth}");
        }

        /// <summary>
        /// Only supported mode is when t1.BatchSize == 1
        /// </summary>
        private static Tensor _differentBatchSizeMultiplication(Tensor t1, Tensor t2)
        {
            if (t1.BatchSize != 1)
                throw new InvalidOperationException($"TENSOR MULTIPLICATION : non supported t1.BatchSize = {t1.BatchSize} != 1 !");

            if (t1.Depth == t2.Depth)
            {
                var t2Offset = 0;
                var result = new Tensor(t2.BatchSize, t1.Depth, t1.Rows, t2.Columns);

                for (int b = 0; b < t2.BatchSize; b++)
                {
                    var t1Offset = 0;
                    for (int d = 0; d < t1.Depth; d++, t1Offset += t1.RowsColumns, t2Offset += t2.RowsColumns)
                        _simpleMatrixMultiplication(b, d, t1Offset, t2Offset, t1, t2, result);
                }

                return result;
            }
            else if (t1.Depth == 1 && t1.Mode == TensorMultiplicationModes.LastLevel)
            {
                var t2Offset = (t2.Depth - 1) * t2.RowsColumns;
                var result = new Tensor(t2.BatchSize, 1, t1.Rows, t2.Columns);


                for (int b = 0; b < t2.BatchSize; b++, t2Offset += t2.DepthRowsColumns)
                    _simpleMatrixMultiplication(b, 0, t1Start : 0, t2Offset, t1, t2, result);

                return result;
            }

            else
                throw new InvalidOperationException($"TENSOR MULTIPLICATION : non supported values of {nameof(t1.Depth)} : {t1.Depth}, {t2.Depth}");
        }

        private static void _simpleMatrixMultiplication(int batch, int depth, int t1Start, int t2Start, Tensor t1, Tensor t2, Tensor result)
        {
            Parallel.For(0, t1.Rows, t1Row =>
            {
                for (int t2Col = 0; t2Col < t2.Columns; t2Col++)
                {
                    var res = 0D;
                    var t1Offset = t1Start + t1Row * t1.Columns;
                    var t2Offset = t2Start + t2Col;

                    // for (int t1Col = 0; t1Col < t1.Columns; t1Col++)
                    //     res += t1[d, t1Row, t1Col] * t2[d, t1Col, t2Col]
                    for (int t1Col = 0; t1Col < t1.Columns; t1Col++, t1Offset++, t2Offset += t2.Columns)
                        res += t1.data[t1Offset] * t2.data[t2Offset];

                    result[batch, depth, t1Row, t2Col] = res;
                }
            });
        }

        public static Tensor operator *(double d1, Tensor t2)
        {
            var result = new Tensor(t2.BatchSize, t2.Depth, t2.Rows, t2.Columns);
            Parallel.For(0, t2.data.Length, i =>
                result.data[i] = d1*t2.data[i]);

            return result;
        }

        public Tensor ApplyFunctionOnAllElements(Func<double, double> func)
        {
            Tensor result = new Tensor(BatchSize, Depth, Rows, Columns);

#if FORCE_PARALLELISM
            int chunk;
            if (result.data.Length > 100)
            {
                chunk = result.data.Length / 4;
                Parallel.For(0, 4, i =>
                {
                    int endIndex = i == 3 ? result.data.Length : chunk * (i + 1);
                    for (int j = i * chunk; j < endIndex; j++)
                        result.data[j] = func(data[j]);
                });
            }
            else
                for (int r = 0; r < data.Length; r++)
                    result.data[r] = func(data[r]);
#else

            for (int r = 0; r < data.Length; r++)
                result.data[r] = func(data[r]);
#endif

            return result;
        }


        public static Tensor operator +(Tensor t1, Tensor t2)
            => elementWiseOp(t1, t2, (x1, x2) => x1 + x2, "ADDITION");

        public static Tensor operator -(Tensor t1, Tensor t2)
            => elementWiseOp(t1, t2, (x1, x2) => x1 - x2, "SUBTRACTION");

        /// <summary>
        /// Column-wise tensor operation between 2 matrices. Just 2 supported modes supposing t1.Rows == t2.Rows : 
        /// 1) t1.Columns == t2.Columns then it's element-wise operation
        /// 2) t2.Columns == 1 then t2[r, 0] is applied to each element in r-th row of m1
        /// </summary>
        private static Tensor elementWiseOp(Tensor t1, Tensor t2, Func<double, double, double> func, string nameOfOperation)
        {
            if (t1.Rows != t2.Rows)
                throw new ArgumentException($"TENSOR {nameOfOperation} : {nameof(t1.Rows)}:{t1.Rows} not equal to {nameof(t2.Rows)}:{t2.Rows} !");

            var result = new Tensor(t1.BatchSize, t1.Depth, t1.Rows, t1.Columns);
            if (t1.Columns == t2.Columns && t1.Depth == t2.Depth && (t1.BatchSize == t2.BatchSize || t2.BatchSize == 1))
            {
                Parallel.For(0, t2.data.Length, i =>
                    result.data[i] = func(t1.data[i], t2.data[(t2.BatchSize == 1) ? i % t2.DepthRowsColumns : i]));

                return result;
            }
            else if (t2.Columns == 1 && t1.Depth == t2.Depth && (t1.BatchSize == t2.BatchSize || t2.BatchSize == 1))
            {
                var b1 = 0;

                for (int b = 0; b < t1.BatchSize; b++, b1 += t1.BatchSize == t2.BatchSize ? 1 : 0)
                    for (int d = 0; d < t1.Depth; d++)
                    {
                        var t1Start = b * t1.DepthRowsColumns + d * t1.RowsColumns;
                        Parallel.For(0, t1.Rows, r =>
                        {
                            int t1Offset = t1Start + r * t1.Columns;
                            var t2Data = t2[b1, d, r, 0];
                            for (int c = 0; c < t1.Columns; c++, t1Offset++)
                                result.data[t1Offset] = func(t1.data[t1Offset], t2Data);
                        });
                    }
                return result;
            }
            else if (t1.BatchSize != t2.BatchSize && t2.BatchSize != 1)
                throw new InvalidOperationException($"{nameOfOperation} : invalid combination of batch sizes : {t1.BatchSize} : {t2.BatchSize}");
            else
                throw new ArgumentException($"MATRIX {nameOfOperation} : NON-SUPPORTED OPERANDS : {nameof(t1.Columns)} : {t1.Columns} ; {nameof(t2.Columns)} : {t2.Columns}");
        }

        public Tensor SumRows()
        {
            if (Columns == 1) // no reference passing, we need a copy
            {
                var res = new Tensor(BatchSize, Depth, Rows, Columns);
                data.ParallelCopyTo(res.data);
                return res;
            }

            var result = new Tensor(BatchSize, Depth, Rows, 1);
            var offset = 0;

            Parallel.For(0, BatchSize, b =>
            {
               for (int d = 0; d < Depth; d++)
                   for (int r = 0; r < Rows; r++)
                   {
                       var res = 0D;
                       for (int c = 0; c < Columns; c++, offset++)
                           res += data[offset];

                       result[b, d, r, 0] = res;
                   }
            });

            return result;
        }

        public Tensor SumBatch()
        {
            if (BatchSize == 1)
            {
                var res = new Tensor(1, Depth, Rows, Columns);
                data.ParallelCopyTo(res.data);
                return res;
            }

            var result = new Tensor(1, Depth, Rows, Columns);
            Parallel.For(1, BatchSize, b => {
                var offset = b * DepthRowsColumns;
                for (int i = 0; i < DepthRowsColumns; i++)
                    result.data[i] = data[i + offset];
            });

            return result;
        }

        /// <summary>
        /// First argument of func is this[indices] and second is auxMatrix[indices]
        /// </summary>
        public Tensor ApplyFunctionOnAllElements(Func<double, double, double> func, Tensor auxMatrix, bool disableChecking = false)
        {
            if (!disableChecking && (auxMatrix.BatchSize != BatchSize || auxMatrix.Depth != Depth || auxMatrix.Rows != Rows || auxMatrix.Columns != Columns))
                throw new InvalidOperationException();

            Tensor result = new Tensor(BatchSize, Depth, Rows, Columns);
#if FORCE_PARALLELISM

            int chunk;
            if (result.data.Length > 100)
            {
                chunk = result.data.Length / 4;
                Parallel.For(0, 4, i =>
                {
                    int endIndex = i == 3 ? result.data.Length : chunk * (i + 1);
                    for (int j = i * chunk; j < endIndex; j++)
                        result.data[j] = func(data[j], auxMatrix.data[j]);
                });
            }
            else
                for (int r = 0; r < data.Length; r++)
                    result.data[r] = func(data[r], auxMatrix.data[r]);
#else

            //for (int r = 0; r < data.Length; r++)
            //    result.data[r] = func(data[r], auxMatrix.data[r]);
#endif

            return result;
        }

        public Tensor ZeroOut()
            => new Tensor(BatchSize, Depth, Rows, Columns);

        public override string ToString()
        {
            StringBuilder builder = new StringBuilder();
            for (int b = 0; b < BatchSize; b++)
            {
                builder.Append($"Batch {b} :\n");
                for (int d = 0; d < Depth; d++)
                {
                    builder.Append($"Depth {d} :\n");
                    for (int r = 0; r < Rows; r++)
                    {
                        for (int c = 0; c < Columns - 1; c++)
                            builder.Append(string.Format("{0,3},", this[b,d, r, c]));
                        builder.Append(string.Format("{0,3}\n", this[b, d, r, Columns - 1]));
                    }
                }
            }
                
           return builder.ToString();
        }
    }

    static class DoubleArrayExtensions
    {
        public static void ParallelCopyTo(this double[] from, double[] to)
        {
            if (from.Length != to.Length)
                throw new ArgumentException("Lenghts not equal, impossible to copy !");
            int chunk;
            if (to.Length > 100)
            {
                chunk = to.Length / 4;
                Parallel.For(0, 4, i =>
                {
                    int endIndex = i == 3 ? to.Length : chunk * (i + 1);
                    for (int j = i * chunk; j < endIndex; j++)
                        to[j] = from[j];
                });
            }
            else
                from.CopyTo(to, 0);
        }

        public static void ParallelCopyTo(this double[][] from, double[] to)
        {
            if (from.Length*from[1].Length != to.Length)
                throw new ArgumentException("Lenghts not equal, impossible to copy !");
            int chunk = from[1].Length;
            Parallel.For(0, from.GetLength(0), i => from[i].CopyTo(to, i * chunk));
        }
    }
}
