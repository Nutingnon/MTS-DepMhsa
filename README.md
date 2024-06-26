# DEP-MHSA: Dual-Enhanced Position Embedding of Multi-Head Self Attention for May-Thurner Syndrome Diagnosis


<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="assets/ps_vs_zq_small.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Samples of CT scans and Enhanced CT scans that collected in this study.</div>
</center>

## Overview
Welcome to the official repository of **DEP-MHSA**, a state-of-the-art deep learning network designed for the accurate diagnosis of May-Thurner Syndrome using advanced self-attention mechanisms.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="assets/mainNetwork.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">The main structure of our proposed network.</div>
</center>


## Features
- **Dual-Enhanced Position Embedding (DEP)**: A unique embedding technique that improves the network's understanding of temporal-spatial relationships in medical images.

- **Expert-like Multi-Head Self-Attention (MHSA)**: The techniques used to generate the Key, Query, and Value matrices in our model draw inspiration directly from medical expertise. The efficacy of this structure has been rigorously validated through a comprehensive ablation study and comparative analysis with networks of similar scale.

- **High Accuracy and Precision**: Rigorously tested on the collected CT scan datasets, which achieved highest performance across F1-Score, Accuracy and AUC.



## Network Structure 
The figure illustrates the main architecture of the DEP-MHSA network. Our model leverages advanced neural network techniques to enhance feature extraction and improve diagnostic accuracy.

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="assets/dep_mhsa_github.png">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">DEP-MHSA and its variants.</div>
</center>


## Getting Started
To use `DEP-MHSA` in your research or clinical practice, follow these steps:

1. Clone the repository: `git clone https://github.com/Nutingnon/MTS-DepMhsa.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Follow the detailed usage instructions in the [Documentation]().

## Data Acquisition
The data will be released once the paper accepted

## Contributing
We welcome contributions to the **DEP-MHSA** project! Please read our [Contributing Guidelines]() for more information on how you can contribute.

## License
**DEP-MHSA** is released under the MIT License.

## Citation
If you use DEP-MHSA or our collected dataset in your research, please cite our paper:

```
@misc{huang2024mtsnet,
      title={MTS-Net: Dual-Enhanced Positional Multi-Head Self-Attention for 3D CT Diagnosis of May-Thurner Syndrome}, 
      author={Yixin Huang and Yiqi Jin and Ke Tao and Kaijian Xia and Jianfeng Gu and Lei Yu and Lan Du and Cunjian Chen},
      year={2024},
      eprint={2406.04680},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
