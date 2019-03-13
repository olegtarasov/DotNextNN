using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DotNextNN.Core;
using Xunit;
using Xunit.Abstractions;

namespace DotNextNN.Tests
{
    public class MatrixTests
    {
        private readonly ITestOutputHelper output;

        public MatrixTests(ITestOutputHelper output)
        {
            this.output = output;
        }

        [Fact]
        public void CompareMklWithLoops()
        {
        }
    }
}
