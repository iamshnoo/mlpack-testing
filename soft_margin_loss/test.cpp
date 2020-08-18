#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

int main()
{
  // Constructor
  arma::mat x,y;

  x << 0.1778 << 0.1203 << -0.2264 << endr
    << 0.0957 << 0.2403 << -0.3400 << endr
    << 0.1397 << 0.1925 << -0.3336 << endr
    << 0.2256 << 0.3144 << -0.8695 << endr;

  y <<  1  <<  1  << -1  << endr
    <<  1  << -1  <<  1  << endr
    << -1  <<  1  <<  1  << endr
    <<  1  <<  1  <<  1  << endr;

  // Forward
  arma::mat loss_none = arma::log(1 + arma::exp(-y % x));
  double loss_sum = arma::sum(arma::sum(loss_none));
  double loss_mean = loss_sum / x.n_elem;

  // Backward
  arma::mat output ;
  output.set_size(size(x));
  arma::mat numerator = -y % arma::exp(-y % x);
  arma::mat denominator = 1 + arma::exp(-y % x);
  output = numerator / denominator;

  // Display
  cout << "------------------------------------------------------------------" << endl;
  cout << "USER-PROVIDED MATRICES : " << endl;
  cout << "------------------------------------------------------------------" << endl;
  cout << "Input shape : "<< x.n_rows << " " << x.n_cols << endl;
  cout << "Input : " << endl << x << endl;
  cout << "Target shape : "<< y.n_rows << " " << y.n_cols << endl;
  cout << "Target : " << endl << y << endl;
  cout << "------------------------------------------------------------------" << endl;
  cout << "SUM " << endl;
  cout << "------------------------------------------------------------------" << endl;
  cout << "FORWARD : " << endl;
  cout << "Loss : \n" << loss_none << '\n';
  cout << "Loss (sum):\n" << loss_sum << '\n';
  cout << "BACKWARD : " << endl;
  cout << "Output shape : "<< output.n_rows << " " << output.n_cols << endl;
  cout << "Output (sum) : " << endl << output << endl;
  cout << "Sum of all values in this matrix : " << arma::as_scalar(arma::accu(output)) << endl;
  cout << "------------------------------------------------------------------" << endl;
  cout << "MEAN " << endl;
  cout << "------------------------------------------------------------------" << endl;
  cout << "FORWARD : " << endl;
  cout << "Loss (mean):\n" << loss_mean << '\n';
  cout << "BACKWARD : " << endl;
  cout << "Output shape : "<< output.n_rows << " " << output.n_cols << endl;
  cout << "Output (mean) : " << endl << output / x.n_elem << endl;
  cout << "Sum of all values in this matrix : " << arma::as_scalar(arma::accu(output / x.n_elem)) << endl;
  cout << "------------------------------------------------------------------" << endl;
  return 0;
}