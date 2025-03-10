#pragma once
// Minimal header-only FFT implementation in C++20
#include <complex>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace fft {

   using Complex = std::complex<double>;
   using ComplexVector = std::vector<Complex>;

   // In-place iterative Cooley-Tukey FFT.
   // Input size must be a power of 2.
   inline void fft_inplace(ComplexVector & a) {
      const size_t n = a.size();
      if(n == 0) return;
      if((n & (n - 1)) != 0)
         throw std::runtime_error("fft_inplace: Input size must be a power of 2.");

      // Bit reversal permutation
      size_t j = 0;
      for (size_t i = 1; i < n; ++i) {
         size_t bit = n >> 1;
         for (; j & bit; bit >>= 1)
            j -= bit;
         j += bit;
         if(i < j)
            std::swap(a[i], a[j]);
      }

      // FFT iterations
      for (size_t len = 2; len <= n; len <<= 1) {
         double angle = -2 * M_PI / static_cast<double>(len);
         Complex wlen(std::cos(angle), std::sin(angle));
         for (size_t i = 0; i < n; i += len) {
            Complex w(1);
            for (size_t j = 0; j < len / 2; ++j) {
               Complex u = a[i + j];
               Complex v = a[i + j + len / 2] * w;
               a[i + j] = u + v;
               a[i + j + len / 2] = u - v;
               w *= wlen;
            }
         }
      }
   }

   // Returns the FFT of the input vector.
   inline ComplexVector fft(const ComplexVector & input) {
      ComplexVector a = input;
      fft_inplace(a);
      return a;
   }

   // Computes the inverse FFT.
   inline ComplexVector ifft(const ComplexVector & input) {
      ComplexVector a = input;
      // Take conjugate of each element.
      for (auto & x : a)
         x = std::conj(x);
      fft_inplace(a);
      // Take conjugate again and scale by 1/n.
      for (auto & x : a)
         x = std::conj(x) / static_cast<double>(a.size());
      return a;
   }

} // namespace fft
