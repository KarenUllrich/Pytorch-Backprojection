##  Differentiable probabilistic models of scientific imaging with the Fourier slice theorem

Pytorch implementation of differntiable **orthogonal integral projection** (backprojection) operator in Fourier space, based on our paper:

[Differentiable probabilistic models of scientific imaging with the Fourier slice theorem](https://arxiv.org/abs/1906.07582)  (UAI 2019)
Karen Ullrich, Rianne van den Berg,  Marcus A. Brubaker, David Fleet, Max Welling

### Requirements

The requirements for the conda environment in which we have tested this code are started in `requirements.txt`.
The main dependencies are 
-   `python 3.6`
-   `pytorch 1.1.0` 

A suitable conda environment may be installed via
	```
	conda env create -f requirements.yml 
	```
And used by
	```
	source activate backprojection
	```
### Usage
We provide a light introduction to scientific imaging in the jupyter-notebook `cryo-tutorial.ipynb`.  Specifically the generative model of scientific imaging `observation_model.py` might prove useful for any application that involves **orthogonal integral projection**.

### Maintenance

Please be warned that this repository is not going to be maintained regularly.

<figure class="video_container">
  <video controls="true" allowfullscreen="true" poster="path/to/poster_image.png">
    <source src="./chimeraX_vis.mp4" type="video/mp4">
  </video>
</figure>

### Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{ullrich2019backprojection,
  title={Differentiable probabilistic models of scientific imaging with the Fourier slice theorem},
  author={Karen Ullrich, Rianne van den Berg,  Marcus A. Brubaker, David Fleet, Max Welling},
  booktitle={proceedings of the Conference on Uncertainty in Artificial Intelligence (UAI)},
  year={2010}
}
```
