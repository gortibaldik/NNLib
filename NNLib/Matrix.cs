using System;
using System.Collections.Generic;
using System.Data.Common;
using System.Text;

namespace NNLib
{
    public class Matrix
    {
        private double[] data;

        public int Rows { get; }
        public int Columns { get; }

        public Matrix(double[,] data)
        {
            Rows = data.GetLength(0);
            Columns = data.GetLength(1);
            this.data = new double[Rows * Columns];

            for (int i = 0; i < Rows; i++)
                for (int o = 0; o < Columns; o++)
                    this.data[i * Columns + o] = data[i,o];
        }

        public Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;
            data = new double[rows*columns];
        }

        public Matrix(int rows, int columns, NInitializer nInit) : this(rows, columns)
        { 
            for (int i = 0; i < data.Length; i++)
                this.data[i] = nInit();
        }

        public double this[int row, int column]
        {
            get => data[row*Columns + column];
            set => data[row*Columns + column] = value;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="m1"></param>
        /// <param name="m2"></param>
        /// <returns></returns>
        public static Matrix operator *(Matrix m1, Matrix m2)
        {
            if (m1.Columns != m2.Rows)
                throw new ArgumentException($"MATRIX MULTIPLICATION : {nameof(m1.Columns)} not equal to {nameof(m2.Rows)} !");

            var result = new Matrix(m1.Rows, m2.Columns);

            for (int m1Row = 0; m1Row < m1.Rows; m1Row++)
                for (int m2Col = 0; m2Col < m2.Columns; m2Col++)
                {
                    var res = 0D;
                    var m1Offset = m1Row * m1.Columns;
                    var m2Offset = m2Col;

                    for (int m1Col = 0; m1Col < m1.Columns; m1Col++, m1Offset++, m2Offset += m2.Columns)
                        res += m1.data[m1Offset] * m2.data[m2Offset];

                    result[m1Row, m2Col] = res;
                }

            return result;
        }

        public static Matrix operator *(double d1, Matrix m2)
        {
            var result = new Matrix(m2.Rows, m2.Columns);
            for (int i = 0; i < m2.data.Length; i++)
                result.data[i] = d1*m2.data[i];

            return result;
        }

        public static Matrix operator +(Matrix m1, Matrix m2)
            => AuxAllElementsOp(m1, m2, (x1, x2) => x1 + x2, "ADDITION");

        public static Matrix operator -(Matrix m1, Matrix m2)
            => AuxAllElementsOp(m1, m2, (x1, x2) => x1 - x2, "SUBTRACTION");

        /// <summary>
        /// Column-wise matrix operation between 2 matrices. Just 2 supported modes supposing m1.Rows == m2.Rows : 
        /// 1) m1.Columns == m2.Columns then it's element-wise operation
        /// 2) m2.Columns == 1 then m2[r, 0] is applied to each element in r-th row of m1
        /// </summary>
        private static Matrix AuxAllElementsOp(Matrix m1, Matrix m2, Func<double, double, double> func, string nameOfOperation)
        {
            if (m1.Rows != m2.Rows)
                throw new ArgumentException($"MATRIX {nameOfOperation} : {nameof(m1.Rows)}:{m1.Rows} not equal to {nameof(m2.Rows)}:{m2.Rows} !");
            if (m1.Columns == m2.Columns)
            {
                var result = new Matrix(m1.Rows, m1.Columns);
                for (int i = 0; i < m1.data.Length; i++)
                    result.data[i] = func(m1.data[i], m2.data[i]);

                return result;
            }
            else if (m2.Columns == 1)
            {
                var result = new Matrix(m1.Rows, m1.Columns);
                for (int i = 0; i < m1.data.Length; i++)
                    result.data[i] = func(m1.data[i], m2.data[i%m1.Columns]);

                return result;
            }
            else
                throw new ArgumentException($"MATRIX {nameOfOperation} : NON-SUPPORTED OPERANDS : {nameof(m1.Columns)} : {m1.Columns} ; {nameof(m2.Columns)} : {m2.Columns}");
        }

        public Matrix Transpose()
        {
            Matrix result = new Matrix(Columns, Rows);
            for (int r = 0; r < Rows; r++)
                for (int c = 0; c < Columns; c++)
                    result[c, r] = this[r, c];

            return result;
        }

        public Matrix ApplyFunctionOnAllElements(Func<double, double> func)
        {
            Matrix result = new Matrix(Rows, Columns);
            for (int r = 0; r < data.Length; r++)
                result.data[r] = func(data[r]);

            return result;
        }

        public Matrix ApplyFunctionOnAllElements(Func<double, double, double> func, Matrix auxMatrix)
        {
            Matrix result = new Matrix(Rows, Columns);
            for (int r = 0; r < data.Length; r++)
                result.data[r] = func(data[r], auxMatrix.data[r]);

            return result;
        }

        // needs to be tested
        public Matrix Reshape(int newRows, int newColumns)
        {
            if (Rows * Columns != newRows * newColumns)
                throw new ArgumentOutOfRangeException($"Cannot reshape {Rows}*{Columns} to {newRows}*{newColumns} !");

            if (newRows < 0 || newColumns < 0)
                throw new ArgumentOutOfRangeException("Dimensions of matrix cannot be less than or equal to zero !");

            var result = new Matrix(newRows, newColumns);
            data.CopyTo(result.data, 0);

            return result;
        }

        public override string ToString()
        {
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < Rows; i++)
            {
                for (int o = 0; o < Columns - 1; o++)
                    builder.Append(string.Format("{0,3},",this[i, o]));
                builder.Append(string.Format("{0,3}\n", this[i, Columns - 1]));
            }
            return builder.ToString();
        }
    }
}
