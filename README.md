# Post-hoc Uncertainty Estimation
<div style="text-align: center;">
    <img src="assets/animation.gif" alt="drawing" width="700"/>
</div>

## Usage
It is suggested to open the repository with PyCharm. Afterwards, install libraries from the `requirements.txt`. For PyCharm usage we suggest to tick _"Emulate terminal in output console"_ for a configuration.

The repository contains 4 main targets to execute.

For the flood segmentation experiments we have the following:
- `example_projects/example_uncertainty_project/source/main.py`
- `example_script/flood_uncertainty.py`

For the label noise experiments we have the following:
- `example_projects/example_label_noise_project/source/main_baseline.py`
- `example_projects/example_label_noise_project/source/main_label_noise.py`

For running the corresponding example projects, mark the `source` directory within `example_uncertainty_project` or `example_label_noise_project` as a source folder and as the working directory for the run configuration.
Then just execute the corresponding listed Python file. Have a look at various execution arguments in the appropriate `utils.py`.

With respect to the example flood segmentation script, execute `flood_uncertainty.py` within the `example_script` folder.

## Datasets
###### Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. 2020. [Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1](https://github.com/cloudtostreet/Sen1Floods11). The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2020, pp. 210-211.
###### International Society for Photogrammetry and Remote Sensing (ISPRS) 2020. [2D Semantic Labeling Contest - Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx). 2022.