<div align="center">
  
# Instructional Video Generation

[Yayuan Li](https://www.linkedin.com/in/yayuan-li-148659272/), [Zhi Cao](zhicao@umich.edu), [Jason J. Corso](https://web.eecs.umich.edu/~jjcorso/)

[COG Research Group, University of Michigan](https://github.com/MichiganCOG)

<a href='https://arxiv.org/abs/2412.04189'><img src='https://img.shields.io/badge/ArXiv-2311.12886-red'></a> 
<a href='https://excitedbutter.github.io/Instructional-Video-Generation/'><img src='https://img.shields.io/badge/Project-Page-Blue'></a>
</div>


We aim to enhance instructional video generation with a diffusion-based framework, achieving state-of-the-art results in hand motion clarity and task-specific region localization.

If you find our project helpful, please give it a star :star: or [cite](#bibtex) it, we would be very grateful :sparkling_heart: .


## Showcases
<table class="center">
  
  <tr>
    <td>Input image</td>
    <td>Result</td>
    <td>Input image</td>
    <td>Result</td>
  </tr>
  
  <tr>
    <td><img src="docs/816.png" alt="Input image" style="width:350px; height:120px"></td>
    <td><img src="docs/816.gif" alt="Result" style="width:350px; height:120px;"></td>
    <td><img src="docs/83.png" alt="Input image" style="width:350px; height:120px;"></td>
    <td><img src="docs/83.gif" alt="Result" style="width:350px; height:120px;"></td>
  </tr>
  
  <tr>
    <td colspan="2" align="center"><strong><em>Action Description:</em></strong> Knit the fabric.</td>
    <td colspan="2" align="center"><strong><em>Action Description:</em></strong> Roll dough.</td>
  </tr>

  <tr>
    <td><img src="docs/36.png" alt="Input image" style="width:350px; height:120px;"></td>
    <td><img src="docs/36.gif" alt="Result" style="width:350px; height:120px;"></td>
    <td><img src="docs/56.png" alt="Input image" style="width:350px; height:120px;"></td>
    <td><img src="docs/56.gif" alt="Result" style="width:350px; height:120px;"></td>
  </tr>
  
  <tr>
    <td colspan="2" style="width:350;" align="center">
      <strong><em>Action Description:</em></strong> Pour vinegar into bowl.
    </td>
    <td colspan="2" style="width:350;" align="center">
      <strong><em>Action Description:</em></strong> Pick up and crack egg.
    </td>
  </tr>
  
</table>


## Framework
![framework](docs/framework.png)

## News 🔥
**2024.12.9**: Released inference code and updated the model to instructional_video_v1.0

## Features Planned
- 💥 Release training code
- 💥 Video generatinon with camera movement.
- 💥 Support Huggingface Demo / Google Colab.
- etc.

## Getting Started
This repository is based on [animate-anything](https://github.com/alibaba/animate-anything).

### Create Conda Environment (Optional)
It is recommended to install Anaconda.

**Windows Installation:** https://docs.anaconda.com/anaconda/install/windows/

**Linux Installation:** https://docs.anaconda.com/anaconda/install/linux/

```bash
conda create -n IVG python=3.10
conda activate IVG
```

### Python Requirements
```bash
pip install -r requirements.txt
```

## Running inference
Please download the [pretrained model](https://drive.google.com/file/d/1sWlr5r54_XxqdgHoCacS7opoucABpEVx/view?usp=drive_link) to output/latent, then run the following command. Please replace the {download_model} to your download model name:
```bash
python train.py --config output/latent/{download_model}/config.yaml --eval validation_data.prompt_image=example/Julienne_carrot.png validation_data.prompt='The person holds a carrot on the chopping board with the left hand and uses a knife in the right hand to julienne the carrot.'
```

To control the motion area, we use the provided script `mask_video.py`. Update the input and output video folder paths as needed, and run the following command:
```bash
python mask_video.py
```

Below are examples of an input image and its corresponding RoM mask:

<p align="center">
<img src="docs/31.png" alt="Original Video Frame" width="45%">
<img src="docs/31_mask.png" alt="Generated RoM Mask" width="45%">
</p>

Then run the following command for inference:
```bash
python train.py --config output/latent/{download_model}/config.yaml --eval validation_data.prompt_image=example/Julienne_carrot.png validation_data.prompt='The person holds a carrot on the chopping board with the left hand and uses a knife in the right hand to julienne the carrot.' validation_data.mask=example/carrot_mask.jpg 
```
<p align="center"> <img src="docs/31.gif" alt="Inference Result" width="60%"> </p>


### Configuration

The configuration uses a YAML config borrowed from [Tune-A-Video](https://github.com/showlab/Tune-A-Video) repositories. 

All configuration details are placed in `example/train_mask_motion.yaml`. Each parameter has a definition for what it does.


## Bibtex
Please cite this paper if you find the code is useful for your research:
```
@misc{li2024instructionalvideogeneration,
      title={Instructional Video Generation}, 
      author={Yayuan Li and Zhi Cao and Jason J. Corso},
      year={2024},
      eprint={2412.04189},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.04189}, 
}
```
