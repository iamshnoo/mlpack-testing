#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main()
{
    /* Constructor */

    arma::mat input;
    /*
    input << 1 << 446 << 42 << arma::endr
          << 2 <<  16 << 63 << arma::endr
          << 3 <<  13 << 63 << arma::endr
          << 4 <<  21 << 21 << arma::endr
          << 1 <<  13 << 11 << arma::endr
          << 32 << 45 << 42 << arma::endr
          << 22 << 16 << 63 << arma::endr
          << 32 << 13 << 42 << arma::endr;
    */
    input << 1  << 19  << arma::endr
          << 2  << 20  << arma::endr
          << 3  << 21  << arma::endr
          << 4  << 22  << arma::endr
          << 5  << 23  << arma:: endr
          << 6  << 24  << arma::endr
          << 7  << 25  << arma::endr
          << 8  << 26  << arma:: endr
          << 9  << 27  << arma::endr
          << 10 << 28  << arma::endr
          << 11 << 29  << arma:: endr
          << 12 << 30  << arma::endr
          << 13 << 31  << arma::endr
          << 14 << 32  << arma:: endr
          << 15 << 33  << arma::endr
          << 16 << 34  << arma::endr
          << 17 << 35  << arma:: endr
          << 18 << 36  << arma::endr;
    cout << "-----------------------------------" << endl;
    cout << "Input shape : " << input.n_rows << " " << input.n_cols << endl;
    cout << "-----------------------------------" << endl;

    const size_t size = 3; // number of channels
    const double eps = 1e-5;
    const double momentum = 0.1;
    arma::mat weights, runningMean, runningVariance, gamma, beta;
    weights.set_size(size + size, 1); // (size + size, 1)
    runningMean.zeros(size, 1); // (size, 1)
    runningVariance.ones(size, 1); // (size, 1)

    /* Reset() */

    gamma = arma::mat(weights.memptr(), size, 1, false, false);  // (size, 1)
    beta = arma::mat(weights.memptr() + gamma.n_elem, size, 1, false, false); // (size, 1)
    gamma.fill(1.0);
    beta.fill(0.0);

    /* Forward */

    // Step-0 : Preparation of temporary cubes for input and output
    const size_t batchSize = input.n_cols;
    const size_t inputSize = input.n_rows / size; // (inputWidth * inputHeight)
    arma::mat output;
    output.set_size(arma::size(input));
    arma::cube inputTemp(const_cast<arma::mat&>(input).memptr(),input.n_rows / size, size, batchSize, false, false);
    arma::cube outputTemp(const_cast<arma::mat&>(output).memptr(),input.n_rows / size, size, input.n_cols, false, false);
    outputTemp = inputTemp; // n_rows = inputSize, n_cols = size, n_slices = batchSize // (4, 2, 3)

    cout << "N     : " << batchSize << endl;
    cout << "C     : " << size << endl;
    cout << "H x W : " << inputSize << endl;
    cout << "-----------------------------------" << endl;
    cout << "Input Cube - " << endl << "each slice is an image of the batch of "<< batchSize << " images" << endl << "each column of a slice is one of the " << size << " channels of the image, HxW is flattened into this single column" << endl;
    cout << "-----------------------------------" << endl;
    inputTemp.print();
    cout << "-----------------------------------" << endl;

    // PURE FORWARD FOR INSTANCE NORM

    arma::cube mean(1, size, batchSize);
    arma::cube variance(1, size, batchSize);
    for (size_t s = 0; s < inputTemp.n_slices; s++)
    {
        arma::mat& currentInputSlice = inputTemp.slice(s);
        arma::mat& currentOutputSlice = outputTemp.slice(s);

        // Step -1 :  Calculate mean and variance
        mean.slice(s) = arma::mean(currentInputSlice,0);
        variance.slice(s) = arma::var(currentInputSlice, 1, 0);

        // Step 2 : Normalisation
        currentOutputSlice -= arma::repmat(mean.slice(s), input.n_rows / size, 1);
        currentOutputSlice /= arma::sqrt(arma::repmat(variance.slice(s), input.n_rows / size, 1) + eps);

        // Step 3 : Scaling
        currentOutputSlice %= arma::repmat(gamma.t(), input.n_rows / size, 1);
        currentOutputSlice += arma::repmat(beta.t(), input.n_rows / size, 1);
    }
    cout << "Input Mean : " << endl;
    cout << mean << endl;
    cout << "-----------------------------------" << endl;
    cout << "Input Variance : " << endl;
    cout << variance << endl;
    cout << "-----------------------------------" << endl;
    cout << "Output Cube - " << endl << "each slice is an image of the batch of "<< batchSize << " images" << endl << "each column of a slice is one of the " << size << " channels of the image, HxW is flattened into this single column" << endl;
    cout << "-----------------------------------" << endl;
    outputTemp.print();
    cout << "-----------------------------------" << endl;

    cout << "Output shape : " << output.n_rows << " " << output.n_cols << endl;
    cout << "-----------------------------------" << endl;

    return 0;
}
