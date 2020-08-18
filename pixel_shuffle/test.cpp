#include <iostream>
#include <armadillo>
#define PRINT 1

using namespace std;
using namespace arma;

int main()
{
  // User input
  arma::mat input;
  input << 1 << 3 << 2 << 4 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << endr
        << 5 << 7 << 6 << 8 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << endr;
  input = input.t();

  size_t size = 4; // input channels
  size_t upscale_factor = 2;
  size_t height = 2;
  size_t width = 2;

  // Forward()
  size_t batchSize = input.n_cols;
  size_t size_out = size / std::pow(upscale_factor, 2); // output channels
  size_t output_height = height * upscale_factor;
  size_t output_width = width * upscale_factor;

  arma::mat output;
  output.zeros(output_height * output_width * size_out, batchSize );

  for(size_t n = 0; n < batchSize; n++)
  {
    arma::mat inputImage = input.col(n);
    arma::mat outputImage = output.col(n);
    arma::cube inputTemp(const_cast<arma::mat&>(inputImage).memptr(), height, width, size, false, false);
    arma::cube outputTemp(const_cast<arma::mat&>(outputImage).memptr(), output_height, output_width, size_out, false, false);

    for (size_t c = 0; c < size_out ; c++)
    {
      for (size_t h = 0; h < output_height; h++)
      {
        for (size_t w = 0; w < output_width; w++)
        {
          size_t height_index = h / upscale_factor;
          size_t width_index = w / upscale_factor;
          size_t channel_index = (upscale_factor * (h % upscale_factor)) + (w % upscale_factor) + (c * std::pow(upscale_factor, 2));
          outputTemp(w, h, c) = inputTemp(width_index, height_index, channel_index);
        }
      }
    }

    output.col(n) = outputImage;

    if(PRINT)
    {
      cout << "Image " << n << " calculations: " << endl;
      cout << "-----------------------------------" << endl;
      cout << "-----------------------------------" << endl;
      cout << "INPUT for Pixel Shuffle: " << endl;
      cout << inputTemp << endl;
      cout << "-----------------------------------" << endl;
      cout << "OUTPUT from Pixel Shuffle: " << endl;
      cout << outputTemp << endl;
      cout << "-----------------------------------" << endl;
    }

  }

  // this is gy for Pixel Shuffle layer simulated as a tensor filled with an arbitrary sequence of values.
  arma::mat gy;
  gy << 1 << 5 << 9 << 13 << 2 << 6 << 10 << 14 << 3 << 7 << 11 << 15 << 4 << 8 << 12 << 16 << endr
    << 17 << 21 << 25 << 29 << 18 << 22 << 26 << 30 << 19 << 23 << 27 << 31 << 20 << 24 << 28 << 32 << endr;
  gy = gy.t();

  // Backward()
  arma::mat g;
  g.zeros(arma::size(input));

  for(size_t n = 0; n < batchSize; n++)
  {
    arma::mat gyImage = gy.col(n);
    arma::mat gImage = g.col(n);
    arma::cube gyTemp(const_cast<arma::mat&>(gyImage).memptr(), output_height, output_width, size_out, false, false);
    arma::cube gTemp(const_cast<arma::mat&>(gImage).memptr(), height, width, size, false, false);

    for (size_t c = 0; c < size_out ; c++)
    {
      for (size_t h = 0; h < output_height; h++)
      {
        for (size_t w = 0; w < output_width; w++)
        {
          size_t height_index = h / upscale_factor;
          size_t width_index = w / upscale_factor;
          size_t channel_index = (upscale_factor * (h % upscale_factor)) + (w % upscale_factor) + (c * std::pow(upscale_factor, 2));
          gTemp(width_index, height_index, channel_index) = gyTemp(w, h, c);
        }
      }
    }

    g.col(n) = gImage;

    if (PRINT)
    {
      cout << "Image " << n << " calculations: " << endl;
      cout << "-----------------------------------" << endl;
      cout << "Hypothetical gy for Pixel Shuffle: " << endl;
      cout << gyTemp << endl;
      cout << "-----------------------------------" << endl;
      cout << "g for Pixel Shuffle: " << endl;
      cout << gTemp << endl;
      cout << "-----------------------------------" << endl;
    }
  }

  return 0;
}
