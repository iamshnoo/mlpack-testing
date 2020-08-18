#include <iostream>
#include <mlpack/methods/ann/dists/bernoulli_distribution.hpp>
#include <armadillo>
#define PRINT 1

using namespace mlpack::ann;
using namespace std;
using namespace arma;

int main()
{
  // Using this only for demo purposes. This will dropout second channel always.
  arma_rng::set_seed(0);

  // User input
  arma::mat input(12, 1);
  input << 0.4963 << 0.0885 << 0.7682 << 0.1320 << 0.3074 << 0.4901 << 0.6341 << 0.8964 << 0.4556 << 0.3489 << 0.6323 << 0.4017 << endr;
  input = input.t();

  double ratio = 0.2;
  size_t size = 3; // input channels

  arma::mat output;
  output.zeros(arma::size(input));

  // Forward()
  size_t batchSize = input.n_cols;
  size_t inputSize = input.n_rows / size;
  double scale = 1.0 / (1.0 - ratio);
  arma::cube inputTemp(const_cast<arma::mat&>(input).memptr(), inputSize, size, batchSize, false, false);
  arma::cube outputTemp(const_cast<arma::mat&>(output).memptr(), inputSize, size, batchSize, false, false);
  arma::mat probabilities(1, size);
  arma::mat maskRow(1, size);
  arma::mat mask;
  probabilities.fill(ratio);
  BernoulliDistribution<> bernoulli_dist(probabilities, false);
  maskRow = bernoulli_dist.Sample();
  mask = arma::repmat(maskRow, inputSize, 1);

  for(size_t n = 0; n < batchSize; n++)
  {
    arma::mat& inputImage = inputTemp.slice(n);
    arma::mat& outputImage = outputTemp.slice(n);
    outputImage = inputImage % mask * scale;

    if(PRINT)
    {
      cout << "Image " << n << " calculations: " << endl;
      cout << "-----------------------------------" << endl;
      cout << "-----------------------------------" << endl;
      cout << "INPUT for Spatial Dropout: " << endl;
      cout << inputImage << endl;
      cout << "-----------------------------------" << endl;
      cout << "OUTPUT for Spatial Dropout: " << endl;
      cout << outputImage << endl;
      cout << "-----------------------------------" << endl;
    }
  }

  // this is gy for Spatial Dropout layer simulated as a tensor filled with an arbitrary sequence of values.
  arma::mat gy;
  gy << 1 << 3 << 2 << 4 << 5 << 7 << 6 << 8 << 9 << 11 << 10 << 12 << endr;
  gy = gy.t();

  // Backward()
  arma::mat g;
  g.zeros(arma::size(input));

  arma::cube gyTemp(const_cast<arma::mat&>(gy).memptr(), inputSize, size, batchSize, false, false);
  arma::cube gTemp(const_cast<arma::mat&>(g).memptr(), inputSize, size, batchSize, false, false);

  for(size_t n = 0; n < batchSize; n++)
  {
    arma::mat& gyImage = gyTemp.slice(n);
    arma::mat& gImage = gTemp.slice(n);

    gImage = gyImage % mask * scale;

    if(PRINT)
    {
      cout << "Image " << n << " calculations: " << endl;
      cout << "-----------------------------------" << endl;
      cout << "-----------------------------------" << endl;
      cout << "Hypothetical gy for Spatial Dropout: " << endl;
      cout << gyImage << endl;
      cout << "-----------------------------------" << endl;
      cout << "g for Spatial Dropout: " << endl;
      cout << gImage << endl;
      cout << "-----------------------------------" << endl;
    }
  }

  return 0;
}
