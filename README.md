# MWTGNN
![mwt](.\Figs\mwt.png)

> Recent studies highlighted that wavelet transform can be organically leveraged to enhance the expressivity of graph neural networks (GNNs). However, these methodologies predominantly rely on the spectral graph wavelet theory, necessitating intricate graph filter design or a meticulously crafted polynomial approximation mechanism. This prompts the inquiry: Is there a simple way to employ wavelet transform in graph representation learning akin to its application in signal processing? To address this question, we propose implementing discrete wavelet transform (DWT) within the wavelet domain and developing a Multilevel Wavelet Transform Graph Neural Network (MWTGNN), which can simultaneously enlarge the receptive field of the network through multilevel decomposition and capture multi-frequency information of the given input. Specifically, the graph signal is decomposed into its constituent low- and high-frequency components using the DWT at each hierarchical level. Subsequently, graph convolution is employed to extract the embedding information pertinent to each level. Following this, the components are reassembled through the inverse DWT. We constantly decompose the raw graph signal for multiple times to fully exploit its multilevel representation and multi-frequency components, for the sake of more discriminative feature aggregation. We evaluate the effectiveness of our framework on various benchmark datasets, and experimental results show that our model not only achieves superior performance compared to existing state-of-the-art methods but also demonstrates computational efficiency.

## Dependencies
- python 3.8.19
- pytorch 1.12.0
- torchvision 0.13.0
- torchaudio 0.12.0
- scikit-learn 1.3.2
- pywavelets==1.4.1

## Easy Start

1. `unzip data.zip -d data`
2. `pip install -r requirement.yaml`
2. `python main`
