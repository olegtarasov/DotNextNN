# C# MNIST example

Extract `minst.zip` to any folder. Run `DotnextNN.ConsoleTest`:

```
./DotNextNN.ConsoleTest TrainMNIST -path=[path to MNIST] -epochs=20
```

You can also run matrix dot product test with

```
./DotNextNN.ConsoleTest CompareBlasWithLoops
```

### OopenBLAS on MacOs

You can install OpenBLAS on MacOs with

```
brew install openblas
```

And you will need to set `LD_LIBRARY_PATH` so that .NET runtime can find the library.

```
export LD_LIBRARY_PATH=$PATH:/usr/local/opt/openblas/lib
```