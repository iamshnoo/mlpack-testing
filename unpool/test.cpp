#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main()
{
  arma::mat output;
  arma::mat input;

  input = arma::zeros(16, 1);
  input(0) = 1.5;
  input(1) = 2.0;
  input(2) = 2.3;
  input(3) = 2.2;
  input(4) = 1.7;
  input(5) = 2.1;
  input(6) = 1.9;
  input(7) = 2.1;
  input(8) = 1.4;
  input(9) = 1.8;
  input(10) = 1.5;
  input(11) = 1.6;
  input(12) = 1.3;
  input(13) = 1.6;
  input(14) = 1.4;
  input(15) = 1.7;

  //input = arma::mat("1.5 2.0 2.3 2.2 1.7 2.1 1.9 2.1 1.4 1.8 1.5 1.6 1.3 1.6 1.4 1.7").t();

  cout << "-------------------------------------" << endl;
  cout << "INPUT : " << endl;
  cout << "Input shape : "<< input.n_rows << " " << input.n_cols << endl;
  cout << "Input (Transposed view) : " << input.t() << endl;
  cout << "-------------------------------------" << endl;

  const size_t kernelWidth = 2;
  const size_t kernelHeight = 2;
  const size_t strideWidth = 2;
  const size_t strideHeight = 2;
  const bool floor = true;
  size_t inSize = 0;
  size_t outSize = 0;
  bool reset = false;
  size_t inputWidth = 0;
  size_t inputHeight = 0;
  size_t outputWidth = 0;
  size_t outputHeight = 0;
  bool deterministic = false;
  size_t offset = 0;
  size_t batchSize = 0;

  inputHeight = 4;
  inputWidth = 4;
  batchSize = input.n_cols;
  inSize = input.n_elem / (inputWidth * inputHeight * batchSize);

  // cube(ptr_aux_mem, n_rows, n_cols, n_slices, copy_aux_mem = true, strict = false)
  arma::cube inputTemp = arma::cube(const_cast<arma::mat &>(input).memptr(),
      inputWidth, inputHeight, batchSize * inSize, false, false);

  cout << "inputTemp BEFORE POOLING: " << endl;
  cout << "inputTemp shape : "<< inputTemp.n_rows << " " << inputTemp.n_cols << endl;
  cout << "Num slices   : " << inputTemp.n_slices << endl;
  cout << inputTemp << endl;
  cout << "-------------------------------------" << endl;

  outputWidth = std::floor(((inputWidth - (double) kernelWidth) / (double) strideWidth) + 1);
  outputHeight = std::floor(((inputHeight - (double) kernelHeight) / (double) strideHeight) + 1);
  offset = 0;

  arma::cube outputTemp = arma::zeros<arma::cube>(outputWidth, outputHeight, batchSize * inSize);

  cout << "outputTemp BEFORE POOLING: " << endl;
  cout << "outputTemp shape : "<< outputTemp.n_rows << " " << outputTemp.n_cols << endl;
  cout << outputTemp << endl;
  cout << "-------------------------------------" << endl;

  std::vector<arma::cube> poolingIndices;
  poolingIndices.push_back(outputTemp);

  arma::Mat<size_t> indices;
  arma::Col<size_t> indicesCol;

  size_t elements = inputWidth * inputHeight;
  indicesCol = arma::linspace<arma::Col<size_t> >(0, (elements - 1), elements);
  indices = arma::Mat<size_t>(indicesCol.memptr(), inputWidth, inputHeight);
  reset = true;

  cout << "All parameters of layer : " << endl;
  cout << "kernelWidth   : " << kernelWidth << endl;
  cout << "kernelHeight  : " << kernelHeight << endl;
  cout << "strideWidth   : " << strideWidth << endl;
  cout << "strideHeight  : " << strideHeight << endl;
  cout << "inputWidth    : " << inputWidth << endl;
  cout << "inputHeight   : " << inputHeight << endl;
  cout << "outputWidth   : " << outputWidth << endl;
  cout << "outputHeight  : " << outputHeight << endl;
  cout << "batchSize     : " << batchSize << endl;
  cout << "inSize        : " << inSize << endl;
  cout << "outSize       : " << batchSize * inSize << endl;
  cout << "offset        : " << offset << endl;
  cout << "-------------------------------------" << endl;

  cout << "Calculations done inside (reset) conditional statement : " << endl;
  cout << "indicesCol : " << endl;
  cout << "indicesCol shape : "<< indicesCol.n_rows << " " << indicesCol.n_cols << endl;
  cout << "indicesCol (Transposed view) : " << indicesCol.t() << endl;
  cout << "indices : " << endl;
  cout << "indices shape : "<< indices.n_rows << " " << indices.n_cols << endl;
  cout << indices << endl;
  cout << "-------------------------------------" << endl;

  for (size_t s = 0; s < inputTemp.n_slices; s++)
  {
    // PoolingOperation(inputTemp.slice(s), outputTemp.slice(s),poolingIndices.back().slice(s));

    const arma::mat& in = inputTemp.slice(s);
    arma::mat& out = outputTemp.slice(s);
    arma::mat& poolIndexes = poolingIndices.back().slice(s);

    cout << "poolingIndices FOR SLICE " << s << " BEFORE POOLING: " << endl;
    cout << poolingIndices.back().slice(s)<< endl;
    cout << "-------------------------------------" << endl;

    cout << "POOLING OPERATIONS START NOW." << endl;
    cout << "-------------------------------------" << endl;

    size_t x = 0;
    for (size_t j = 0, colidx = 0; j < out.n_cols; ++j, colidx += strideHeight)
    {
      size_t y = 0;
      for (size_t i = 0, rowidx = 0; i < out.n_rows; ++i, rowidx += strideWidth)
      {

        arma::mat subInput = in(
            arma::span(rowidx, rowidx + kernelWidth - 1 - offset),
            arma::span(colidx, colidx + kernelHeight - 1 - offset)
        );

        // const size_t idx = pooling.Pooling(subInput);
        const size_t idx = arma::as_scalar(arma::find(subInput.max() == subInput, 1));
        out(i, j) = subInput(idx);


        arma::Mat<size_t> subIndices = indices(
            arma::span(rowidx, rowidx + kernelWidth - 1 - offset),
            arma::span(colidx, colidx + kernelHeight - 1 - offset)
        );

        poolIndexes(i, j) = subIndices(idx);

        cout << "COUNTER (" << x << ", " << y << ") CALCULATIONS : " << endl;
        cout << "subInput : " << endl;
        cout << subInput << endl;
        cout << "subIndices : " << endl;
        cout << subIndices << endl;
        cout << "idx : " << idx << endl;
        cout << "subInput(idx) : " << endl;
        cout << subInput(idx) << endl;
        cout << "subIndices(idx) : " << endl;
        cout << subIndices(idx) << endl;
        cout << "-------------------------------------" << endl;
        y++;
      }
      x++;
    }

  cout << "POOLING OPERATIONS END NOW." << endl;
  cout << "-------------------------------------" << endl;

  cout << "poolingIndices FOR SLICE " << s << " POST POOLING: " << endl;
  cout << poolingIndices.back().slice(s)<< endl;
  cout << "-------------------------------------" << endl;

  }

  cout << "inputTemp AFTER POOLING: " << endl;
  cout << "inputTemp shape : "<< inputTemp.n_rows << " " << inputTemp.n_cols << endl;
  cout << "Num slices   : " << inputTemp.n_slices << endl;
  cout << inputTemp << endl;
  cout << "-------------------------------------" << endl;

  cout << "outputTemp AFTER POOLING: " << endl;
  cout << "outputTemp shape : "<< outputTemp.n_rows << " " << outputTemp.n_cols << endl;
  cout << outputTemp << endl;
  cout << "-------------------------------------" << endl;

  output = arma::mat(outputTemp.memptr(), outputTemp.n_elem / batchSize, batchSize);
  outputWidth = outputTemp.n_rows;
  outputHeight = outputTemp.n_cols;
  outSize = batchSize * inSize;

  cout << "OUTPUT : " << endl;
  cout << "Output shape : "<< output.n_rows << " " << output.n_cols << endl;
  cout << "Output (Transposed view) : " << output.t() << endl;
  cout << "-------------------------------------" << endl;


  cout << "-------------------------------------" << endl;
  cout << "PSEUDO BACKWARD : " << endl;
  cout << "-------------------------------------" << endl;

  arma::cube gTemp;
  gTemp = arma::zeros<arma::cube>(inputTemp.n_rows, inputTemp.n_cols, inputTemp.n_slices);

  arma::cube mappedError;
  mappedError = arma::ones<arma::cube>(outputWidth, outputHeight, outSize);

  for (size_t s = 0; s < mappedError.n_slices; s++)
  {
    //Unpooling(mappedError.slice(s), gTemp.slice(s), poolingIndices.back().slice(s));

    const arma::mat& error = mappedError.slice(s);
    arma::mat& out = gTemp.slice(s);
    arma::mat& poIdx = poolingIndices.back().slice(s);

    for (size_t i = 0; i < poIdx.n_elem; ++i)
    {
      out(poIdx(i)) += error(i);
    }

    cout << "OUTPUT : " << endl;
    cout << "Output shape : "<< out.n_rows << " " << out.n_cols << endl;
    cout << "Output (Transposed view) : " << endl << out << endl;
    cout << "-------------------------------------" << endl;
  }


  cout << "-------------------------------------" << endl;
  cout << "UNPOOLING : " << endl;
  cout << "-------------------------------------" << endl;

  // Output of MaxPool is the input for UnPooling.
  input = arma::zeros(4, 1);
  input(0) = 2.1;
  input(1) = 2.3;
  input(2) = 1.8;
  input(3) = 1.7;

  cout << "-------------------------------------" << endl;
  cout << "INPUT FOR UNPOOLING: " << endl;
  cout << "Input shape : "<< input.n_rows << " " << input.n_cols << endl;
  cout << "Input (Transposed view) : " << input.t() << endl;
  cout << "-------------------------------------" << endl;

  inSize = 0;
  outSize = 0;
  reset = false;
  inputWidth = 0;
  inputHeight = 0;
  outputWidth = 0;
  outputHeight = 0;
  batchSize = 0;

  inputHeight = 2;
  inputWidth = 2;
  batchSize = input.n_cols;
  inSize = input.n_elem / (inputWidth * inputHeight * batchSize);

  // cube(ptr_aux_mem, n_rows, n_cols, n_slices, copy_aux_mem = true, strict = false)
  inputTemp = arma::cube(const_cast<arma::mat &>(input).memptr(),
      inputWidth, inputHeight, batchSize * inSize, false, false);

  cout << "inputTemp BEFORE UNPOOLING: " << endl;
  cout << "inputTemp shape : "<< inputTemp.n_rows << " " << inputTemp.n_cols << endl;
  cout << "Num slices   : " << inputTemp.n_slices << endl;
  cout << inputTemp << endl;
  cout << "-------------------------------------" << endl;

  outputWidth = (inputWidth - 1) * strideWidth + kernelWidth;
  outputHeight = (inputHeight - 1) * strideHeight + kernelHeight;
  outSize = batchSize * inSize;
  outputTemp = arma::zeros<arma::cube>(outputWidth, outputHeight,
      outSize);

  cout << "outputTemp BEFORE UNPOOLING: " << endl;
  cout << "outputTemp shape : "<< outputTemp.n_rows << " " << outputTemp.n_cols << endl;
  cout << outputTemp << endl;
  cout << "-------------------------------------" << endl;

  for (size_t s = 0; s < inputTemp.n_slices; s++)
  {
    // Unpooling(inputTemp.slice(s), outputTemp.slice(s), poolingIndices.back().slice(s));
    const arma::mat& IN = inputTemp.slice(s);
    arma::mat& OUT = outputTemp.slice(s);
    arma::mat& INDICES = poolingIndices.back().slice(s);

    for (size_t i = 0; i < INDICES.n_elem; ++i)
    {
      OUT(INDICES(i)) += IN(i);
    }

    cout << "OUTPUT : " << endl;
    cout << "Output shape : "<< OUT.n_rows << " " << OUT.n_cols << endl;
    cout << "Output : " << endl << OUT << endl;
    cout << "-------------------------------------" << endl;
  }

  cout << "inputTemp AFTER UNPOOLING: " << endl;
  cout << "inputTemp shape : "<< inputTemp.n_rows << " " << inputTemp.n_cols << endl;
  cout << "Num slices   : " << inputTemp.n_slices << endl;
  cout << inputTemp << endl;
  cout << "-------------------------------------" << endl;

  cout << "outputTemp AFTER UNPOOLING: " << endl;
  cout << "outputTemp shape : "<< outputTemp.n_rows << " " << outputTemp.n_cols << endl;
  cout << outputTemp << endl;
  cout << "-------------------------------------" << endl;

  cout << "-------------------------------------" << endl;
  cout << "UNPOOLING BACKWARD (PROOF OF CONCEPT): " << endl;
  cout << "-------------------------------------" << endl;

  arma::mat L;
  L = arma::zeros(16, 1);
  L(0) = 1;
  L(1) = 2;
  L(2) = 3;
  L(3) = 4;
  L(4) = 5;
  L(5) = 6;
  L(6) = 7;
  L(7) = 8;
  L(8) = 9;
  L(9) = 10;
  L(10) = 11;
  L(11) = 12;
  L(12) = 13;
  L(13) = 14;
  L(14) = 15;
  L(15) = 16;

  mappedError = arma::cube(const_cast<arma::mat &>(L).memptr(), outputWidth, outputHeight, outSize, false, false);
  gTemp = arma::zeros<arma::cube>(inputWidth, inputHeight, outSize);

  for (size_t s = 0; s < gTemp.n_slices; s++)
  {
    arma::mat& gySlice = mappedError.slice(s);
    arma::mat& gSlice = gTemp.slice(s);
    arma::mat& idxs = poolingIndices.back().slice(s);

    cout << "gySlice: " << endl;
    cout << "gy shape : "<< gySlice.n_rows << " " << gySlice.n_cols << endl;
    cout << gySlice << endl;
    cout << "-------------------------------------" << endl;

    cout << "gTempSlice BEFORE: " << endl;
    cout << "g shape : "<< gSlice.n_rows << " " << gSlice.n_cols << endl;
    cout << gSlice << endl;
    cout << "-------------------------------------" << endl;

    for (size_t i = 0, j = 0; i < idxs.n_elem; ++i, ++j)
    {
      gSlice(j) = gySlice(idxs(i));
    }

    cout << "gTempSlice AFTER: " << endl;
    cout << "g shape : "<< gSlice.n_rows << " " << gSlice.n_cols << endl;
    cout << gSlice << endl;
    cout << "-------------------------------------" << endl;
  }

  arma::mat g = arma::mat(gTemp.memptr(), gTemp.n_elem / batchSize, batchSize);
  cout << "g: " << endl;
  cout << "g shape : "<< g.n_rows << " " << g.n_cols << endl;
  cout << g << endl;
  cout << "-------------------------------------" << endl;

  return 0;
}
