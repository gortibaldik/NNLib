using System;
using System.Collections.Generic;
using System.Text;

namespace NNLib
{
    public delegate double NInitializer();
    public static class NeuronInitializers
    {
        private static Random _RND;
        static NeuronInitializers()
        {
            _RND = new Random();
        }

        public static double NInitZero()
            => 0;

        public static double NInitOne()
            => 1;

        public static double NInitNormal()
        {
            double x1 = 1.0 - _RND.NextDouble();
            double x2 = 1.0 - _RND.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(x1)) * Math.Sin(2.0 * Math.PI * x2);
        }

        public static double NInitGlorotUniform(int rows, int columns)
        {
            var max = 6.0 / (rows + columns);
            return _RND.NextDouble() * 2 * max - max;
        }
    }
}
