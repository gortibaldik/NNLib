using System;
using System.Text;

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

        public int Depth { get; }
        public int Rows { get; }
        public int Columns { get; }
        public TensorMultiplicationModes Mode { get; set; } = TensorMultiplicationModes.OnlySameDepth;


        private int rowsColumns = 0;

        public Tensor(int depth, int rows, int columns, double[] data)
        {
            if (data.Length != depth * rows * columns)
                throw new ArgumentException($"array {nameof(data)} doesn't contain enough elements to fill {depth}x{rows}x{columns} tensor !");

            Depth = depth;
            Rows = rows;
            Columns = columns;
            rowsColumns = rows * columns;
            data.CopyTo(this.data, 0);
        }

        public Tensor(double[,,] data)
        {
            Depth = data.GetLength(0);
            Rows = data.GetLength(1);
            Columns = data.GetLength(2);
            rowsColumns = Rows * Columns;
            this.data = new double[Depth * Rows * Columns];

            for (int d = 0; d < Depth; d++)
                for (int r = 0; r < Rows; r++)
                    for (int c = 0; c < Columns; c++)
                        this[d,r,c] = data[d,r,c];
        }

        public Tensor(int depth, int rows, int columns)
        {
            Rows = rows;
            Columns = columns;
            Depth = depth;
            rowsColumns = rows * columns;
            data = new double[depth*rows*columns];
        }

        public Tensor(int depth, int rows, int columns, NInitializer nInit) : this(depth, rows, columns)
        { 
            for (int i = 0; i < data.Length; i++)
                this.data[i] = nInit();
        }

        public double this[int depth, int row, int column]
        {
            get => data[depth*rowsColumns + row*Columns + column];
            set => data[depth*rowsColumns + row*Columns + column] = value;
        }


        public static Tensor operator *(Tensor t1, Tensor t2)
        {
            if (t1.Columns != t2.Rows)
                throw new ArgumentException($"TENSOR MULTIPLICATION : {nameof(t1.Columns)} not equal to {nameof(t2.Rows)} !");

            if (t1.Depth == t2.Depth)
                return multiplicationSameDepth(t1, t2);

            else if (t1.Depth == 1 && t1.Mode == TensorMultiplicationModes.LastLevel)
                return multiplicationLastDimension(t1, t2);

            else
                throw new InvalidOperationException($"TENSOR MULTIPLICATION : non supported values of {nameof(t1.Depth)} : {t1.Depth}, {t2.Depth}");
        }

        /// <summary>
        /// Supposing t1.Depth == t2.Depth && t1.Columns == t2.Rows
        /// --- !!! checking should be done before calling the method !!! ---
        /// Each level of depth is treated like a distinct matrix multiplication
        /// D x R1 x C1 * D x R2 x C2 = D x R1 x C2
        /// </summary>
        private static Tensor multiplicationSameDepth(Tensor t1, Tensor t2)
        {
            var t1DepthOffset = 0;
            var t2DepthOffset = 0;
            var result = new Tensor(t1.Depth, t1.Rows, t2.Columns);

            for (int d = 0; d < t1.Depth; d++, t1DepthOffset += t1.rowsColumns, t2DepthOffset += t2.rowsColumns)
                for (int t1Row = 0; t1Row < t1.Rows; t1Row++)
                    for (int t2Col = 0; t2Col < t2.Columns; t2Col++)
                    {
                        var res = 0D;
                        var t1Offset = t1DepthOffset + t1Row * t1.Columns;
                        var t2Offset = t2DepthOffset + t2Col;

                        // for (int t1Col = 0; t1Col < t1.Columns; t1Col++)
                        //     res += t1[d, t1Row, t1Col] * t2[d, t1Col, t2Col]
                        for (int t1Col = 0; t1Col < t1.Columns; t1Col++, t1Offset++, t2Offset += t2.Columns)
                            res += t1.data[t1Offset] * t2.data[t2Offset];

                        result[d, t1Row, t2Col] = res;
                    }

            return result;
        }

        /// <summary>
        /// Supposing t1.Depth == 1 && t1.Columns == t2.Rows
        /// --- !!! checking should be done before calling the method !!! ---
        /// Last level of depth of t2 is multiplied with the only level of depth of t1
        /// 1 x R1 x C1 * D x R2 x C2 = 1 x R1 x C2
        /// </summary>
        private static Tensor multiplicationLastDimension(Tensor t1, Tensor t2)
        {
            var t2DepthOffset = (t2.Depth - 1) * t2.rowsColumns;
            var result = new Tensor(1, t1.Rows, t2.Columns);

            for (int t1Row = 0; t1Row < t1.Rows; t1Row++)
                for (int t2Col = 0; t2Col < t2.Columns; t2Col++)
                {
                    var res = 0D;
                    var t1Offset = t1Row * t1.Columns;
                    var t2Offset = t2DepthOffset + t2Col;

                    // for (int t1Col = 0; t1Col < t1.Columns; t1Col++)
                    //     res += t1[d, t1Row, t1Col] * t2[d, t1Col, t2Col]
                    for (int t1Col = 0; t1Col < t1.Columns; t1Col++, t1Offset++, t2Offset += t2.Columns)
                        res += t1.data[t1Offset] * t2.data[t2Offset];

                    result[0, t1Row, t2Col] = res;
                }

            return result;
        }

        public static Tensor operator *(double d1, Tensor t2)
        {
            var result = new Tensor(t2.Depth, t2.Rows, t2.Columns);
            for (int i = 0; i < t2.data.Length; i++)
                result.data[i] = d1*t2.data[i];

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
            if (t1.Columns == t2.Columns && t1.Depth == t2.Depth)
            {
                var result = new Tensor(t1.Depth, t1.Rows, t1.Columns);
                for (int i = 0; i < t1.data.Length; i++)
                    result.data[i] = func(t1.data[i], t2.data[i]);

                return result;
            }
            else if (t2.Columns == 1 && t1.Depth == t2.Depth)
            {
                var result = new Tensor(t1.Depth, t1.Rows, t1.Columns);

                for (int d = 0; d < t1.Depth; d++)
                    for (int r = 0; r < t1.Rows; r++)
                    {
                        int t1Offset = d * t1.rowsColumns + r * t1.Columns;
                        var t2Data = t2[d, r, 0];
                        for (int c = 0; c < t1.Columns; c++, t1Offset++)
                            result.data[t1Offset] = func(t1.data[t1Offset], t2Data);
                    }

                return result;
            }
            else
                throw new ArgumentException($"MATRIX {nameOfOperation} : NON-SUPPORTED OPERANDS : {nameof(t1.Columns)} : {t1.Columns} ; {nameof(t2.Columns)} : {t2.Columns}");
        }

        public Tensor Transpose()
        {
            Tensor result = new Tensor(Depth, Columns, Rows);
            for (int d = 0; d < Depth; d++)
                for (int r = 0; r < Rows; r++)
                    for (int c = 0; c < Columns; c++)
                        result[d, c, r] = this[d, r, c];

            return result;
        }

        public Tensor ApplyFunctionOnAllElements(Func<double, double> func)
        {
            Tensor result = new Tensor(Depth, Rows, Columns);
            for (int r = 0; r < data.Length; r++)
                result.data[r] = func(data[r]);

            return result;
        }

        public Tensor ApplyFunctionOnAllElements(Func<double, double, double> func, Tensor auxMatrix)
        {
            Tensor result = new Tensor(Depth, Rows, Columns);
            for (int r = 0; r < data.Length; r++)
                result.data[r] = func(data[r], auxMatrix.data[r]);

            return result;
        }
 
        public Tensor Reshape(int newDepth, int newRows, int newColumns)
        {
            if (Depth * Rows * Columns != newRows * newColumns * newDepth)
                throw new ArgumentOutOfRangeException($"Cannot reshape {Rows}*{Columns} to {newRows}*{newColumns} !");

            if (newRows < 0 || newColumns < 0 || newDepth < 0)
                throw new ArgumentOutOfRangeException("Dimensions of matrix cannot be less than or equal to zero !");

            var result = new Tensor(newDepth, newRows, newColumns);
            data.CopyTo(result.data, 0);

            return result;
        }

        public Tensor SumRows()
        {
            if (Columns == 1) // no reference passing, we need a copy
            {
                var res = new Tensor(Depth, Rows, 1);
                data.CopyTo(res.data, 0);
                return res;
            }

            var result = new Tensor(Depth, Rows, 1);
            var offset = 0;

            for (int d = 0; d < Depth; d++)
                for (int r = 0; r < Rows; r++)
                {
                    var res = 0D;
                    for (int c = 0; c < Columns; c++, offset++)
                        res += data[offset];

                    result[d, r, 0] = res;
                }

            return result;
        }

        public override string ToString()
        {
            StringBuilder builder = new StringBuilder();
            for (int d = 0; d < Depth; d++)
            {
                builder.Append($"Depth {d} :\n");
                for (int r = 0; r < Rows; r++)
                {
                    for (int c = 0; c < Columns - 1; c++)
                        builder.Append(string.Format("{0,3},", this[d, r, c]));
                    builder.Append(string.Format("{0,3}\n", this[d, r, Columns - 1]));
                }
            }
            return builder.ToString();
        }
    }
}
