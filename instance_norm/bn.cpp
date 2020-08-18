#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main()
{
  ///////////////////////////ANN LAYER TEST (USER INPUT)////////////////////////
  arma::mat input;
  input << 1  << 19  << arma::endr
        << 2  << 20  << arma::endr
        << 3  << 21  << arma::endr
        << 4  << 22  << arma::endr
        << 5  << 23  << arma::endr
        << 6  << 24  << arma::endr
        << 7  << 25  << arma::endr
        << 8  << 26  << arma::endr
        << 9  << 27  << arma::endr
        << 10 << 28  << arma::endr
        << 11 << 29  << arma::endr
        << 12 << 30  << arma::endr
        << 13 << 31  << arma::endr
        << 14 << 32  << arma::endr
        << 15 << 33  << arma::endr
        << 16 << 34  << arma::endr
        << 17 << 35  << arma::endr
        << 18 << 36  << arma::endr;

  size_t size = 3; // number of channels
  const double eps = 1e-5;
  const double momentum = 0.1;
  //////////////////////////////////////////////////////////////////////////////

  ///////////////////////////INSTANCE NORM FORWARD//////////////////////////////
  const size_t shapeA = input.n_rows;
  const size_t shapeB = input.n_cols;
  const size_t shapeC = size;
  arma::mat runningTemp = arma::zeros(shapeC, 1);
  size *= input.n_cols;
  input = arma::vectorise(input);

    /////////////////////////BATCH NORM RESET + FORWARD/////////////////////////
    arma::mat weights, runningMean, runningVariance;
    weights.set_size(size + size, 1);
    runningMean.zeros(size, 1);
    runningVariance.ones(size, 1);
    arma::mat gamma, beta;
    gamma = arma::mat(weights.memptr(), size, 1, false, false);
    beta = arma::mat(weights.memptr() + gamma.n_elem, size, 1, false, false);
    gamma.fill(1.0);
    beta.fill(0.0);
    const size_t batchSize = input.n_cols;
    const size_t inputSize = input.n_rows / size;
    arma::mat output;
    output.set_size(arma::size(input));
    arma::cube inputTemp(const_cast<arma::mat&>(input).memptr(), inputSize, size, batchSize, false, false);
    arma::cube outputTemp(const_cast<arma::mat&>(output).memptr(), inputSize, size, input.n_cols, false, false);
    outputTemp = inputTemp;
    arma::mat mean = arma::mean(arma::mean(inputTemp, 2), 0);
    arma::mat variance = arma::mean(arma::mean(arma::pow(inputTemp.each_slice() - arma::repmat(mean,inputSize, 1), 2), 2), 0);
    outputTemp.each_slice() -= arma::repmat(mean, inputSize, 1);
    arma::cube inputMean;
    inputMean.set_size(arma::size(inputTemp));
    inputMean = outputTemp;
    outputTemp.each_slice() /= arma::sqrt(arma::repmat(variance, inputSize, 1) + eps);
    arma::cube normalized;
    normalized.set_size(arma::size(inputTemp));
    normalized = outputTemp;
    outputTemp.each_slice() %= arma::repmat(gamma.t(),inputSize, 1);
    outputTemp.each_slice() += arma::repmat(beta.t(), inputSize, 1);
    double nElements = 1.0 / (input.n_elem - size + eps);
    runningMean = (1 - momentum) * runningMean + momentum * mean.t();
    runningVariance = (1 - momentum) * runningVariance + input.n_elem * nElements * momentum * variance.t();
    //////////////////////////////////////////////////////////////////////////////

  input.reshape(shapeA, shapeB);
  output.reshape(shapeA, shapeB);
  runningMean.reshape(shapeC, shapeB);
  runningVariance.reshape(shapeC, shapeB);
  runningTemp = arma::mean(runningMean, 1);
  runningMean.set_size(shapeC, 1);
  runningMean = runningTemp;
  runningTemp = arma::mean(runningVariance, 1);
  runningVariance.set_size(shapeC, 1);
  runningVariance = runningTemp;
  mean.reshape(shapeC, shapeB);
  //////////////////////////////////////////////////////////////////////////////

  ///////////////////////////ANN LAYER TEST (USER INPUT)////////////////////////
  arma::mat gy = output;
  //////////////////////////////////////////////////////////////////////////////

  ///////////////////////////INSTANCE NORM BACKWARD/////////////////////////////
  gy = arma::vectorise(gy);
  input = arma::vectorise(input);

    ///////////////////////////BATCH NORM BACKWARD////////////////////////////////
    arma::mat g;
    const arma::mat stdInv = 1.0 / arma::sqrt(variance + eps);
    g.set_size(arma::size(input));
    arma::cube gyTemp(const_cast<arma::mat&>(gy).memptr(), input.n_rows / size, size, input.n_cols, false, false);
    arma::cube gTemp(const_cast<arma::mat&>(g).memptr(), input.n_rows / size, size, input.n_cols, false, false);
    arma::cube norm = gyTemp.each_slice() % arma::repmat(gamma.t(), input.n_rows / size, 1);
    arma::mat temp = arma::sum(norm % inputMean, 2);
    arma::mat vars = temp % arma::repmat(arma::pow(stdInv, 3), input.n_rows / size, 1) * -0.5;
    gTemp = (norm.each_slice() % arma::repmat(stdInv, input.n_rows / size, 1) + (inputMean.each_slice() % vars * 2)) / input.n_cols;
    arma::mat normTemp = arma::sum(norm.each_slice() %arma::repmat(-stdInv, input.n_rows / size, 1) , 2) / input.n_cols;
    gTemp.each_slice() += normTemp;
    //////////////////////////////////////////////////////////////////////////////

  input.reshape(shapeA, shapeB);
  output.reshape(shapeA, shapeB);
  g.reshape(shapeA, shapeB);
  gy.reshape(shapeA, shapeB);
  //////////////////////////////////////////////////////////////////////////////

  cout << "-----------------------------------" << endl;
  mean.print("Input mean: ");
  cout << "-----------------------------------" << endl;
  variance.print("Input variance: ");
  cout << "-----------------------------------" << endl;
  output.print("Output: ");
  cout << "-----------------------------------" << endl;
  runningMean.print("Running Mean: ");
  cout << "-----------------------------------" << endl;
  runningVariance.print("Running Variance: ");
  cout << "-----------------------------------" << endl;
  g.print("g: ");
  cout << "-----------------------------------" << endl;
  gy.print("gy: ");
  cout << "-----------------------------------" << endl;
  cout << "Sum of values in g matrix : " << arma::accu(g) << endl;
  cout << "-----------------------------------" << endl;

  return 0;
}
