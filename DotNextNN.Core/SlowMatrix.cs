using System.Reflection.Emit;
using Retia.RandomGenerator;

namespace DotNextNN.Core
{
    public class SlowMatrix
    {
        public static float[][] FromMatrix(Matrix matrix)
        {
            var columns = new float[matrix.Cols][];
            var storage = (float[])matrix;
            int idx = -1;
            for (int col = 0; col < matrix.Cols; col++)
            {
                var column = new float[matrix.Rows];
                for (int row = 0; row < matrix.Rows; row++)
                {
                    column[row] = storage[++idx];
                }

                columns[col] = column;
            }

            var result = new float[matrix.Rows][];
            for (int row = 0; row < matrix.Rows; row++)
            {
                var newRow = new float[matrix.Cols];
                for (int col = 0; col < matrix.Cols; col++)
                {
                    newRow[col] = columns[col][row];
                }

                result[row] = newRow;
            }

            return result;
        }

        public static float[][] DotProduct(float[][] a, float[][] b)
        {
            var c = new float[a.Length][];
            for (int i = 0; i < a.Length; i++)
            {
                c[i] = new float[b[0].Length];
                for (int j = 0; j < b[0].Length; j++)
                {
                    float cell = 0.0f;
                    for (int k = 0; k < a[0].Length; k++)
                        cell += a[i][k] * b[k][j];

                    c[i][j] = cell;
                }
            }

            return c;
        }
    }
}