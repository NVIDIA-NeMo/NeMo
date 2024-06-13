## Required Dependencies for Hyena

We depend on 3rd-party libraries for FFT convolutions implementation. Each library supports different use-cases:

|     Library      | Supported Sequence Length | Single/Multi-Head Support |
|:----------------:|:-------------------------:|:-------------------------:|
| Safari `fftconv` |        Up to 8192         |       1 or 8 heads        |
|   FlashFFTConv   |         Up to 4M          |     Single-head only      |

Note the overlapping support for single-head with sequence length up to 8192. By default, in this case we default to Safari `fftconv` as it is faster (and fallback to FlashFFTConv). The user may force the FFT convolution implementation used by setting the configuration key `model.hyena.fftconv_type` to either `safari` or `flash`.

### Installation

#### Safari `fftconv`

Install from the [Safari repository](https://github.com/HazyResearch/safari/tree/main/csrc/fftconv). Run the following in a terminal:

```bash
git clone https://github.com/HazyResearch/safari.git
cd safari/csrc/fftconv
pip install .
```

#### FlashFFTConv

Follow the [installation instructions](https://github.com/HazyResearch/flash-fft-conv?tab=readme-ov-file#installation) in the FlashFFTConv repository.
