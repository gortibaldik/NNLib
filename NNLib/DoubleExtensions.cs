using System;
using System.Threading.Tasks;

namespace NNLib
{
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
            if (from.Length * from[0].Length != to.Length)
                throw new ArgumentException("Lenghts not equal, impossible to copy !");
            int chunk = from[0].Length;
            Parallel.For(0, from.GetLength(0), i => from[i].CopyTo(to, i * chunk));
        }
    }
}
