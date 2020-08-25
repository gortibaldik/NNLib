using System;
using System.Threading.Tasks;

namespace NNLib
{
    static class DoubleArrayExtensions
    {
        /// <summary>
        /// Copies the content of this to the specified array in 4 parallel chunks.
        /// </summary>
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

        /// <summary>
        /// Copies the content of this to the specified array : new[i+j] = old[i][j]
        /// Assumes that from[i].Length == from[j].Length for all possible i,j
        /// Assumes that there is exactly enough space to fill all the elements of from to to 
        /// If any of assumptions is broken, throws ArgumentException.
        /// </summary>
        public static void ParallelCopyTo(this double[][] from, double[] to)
        {
            if (from.Length * from[0].Length != to.Length)
                throw new ArgumentException("Lenghts not equal, impossible to copy !");
            int chunk = from[0].Length;
            bool corrupted = false;
            Parallel.For(0, from.GetLength(0), i => {
                if (from[i].Length == chunk)
                    from[i].CopyTo(to, i * chunk);
                else 
                    corrupted = true;
            });

            if (corrupted)
                throw new ArgumentException("The sizes of arrays in this aren't of the same lengths !");
        }
    }
}
