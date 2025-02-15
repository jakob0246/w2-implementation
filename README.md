# W² - How certain and why? Uncertainty estimation and eXplainable AI for DLR

## Background
Artificial intelligence (AI) systems are used in various forms in many DLR 
institutes because they are often the most powerful automated prediction 
systems. Standardized software libraries, especially for deep neural networks, 
make it easier to access such models and allow corresponding systems to be 
trained relatively easily with large amounts of data and to be applied to o
ne's own tasks. Although deep neural networks often achieve excellent results, 
they are usually handled as black-box systems, i.e. details of the function 
that maps input data to target variables are unknown. The limitations of such 
a system can only be evaluated through extensive testing. But even then, 
important questions remain unanswered. This has far-reaching consequences 
for the safeguarding and trustworthiness of the results. Transparent functional 
principles ensure that the results can be assigned to a clear phenomenological 
context and that the connection to physical laws can be maintained. 
However, information suitable for better understanding and interpreting the model 
and its decisions is not provided by most AI systems. 

## Aim
The aim of the project 
is the development and testing of tools that allow analyzing and interpreting 
pre-trained models. This allows the reliability of the results to be assessed and 
increased, which supports broad application inside and outside the DLR. The tools 
are evaluated using the example of current research topics (surface reconstruction, 
automatic detection, flood monitoring, crack detection and tomography segmentation) 
and also include software modules that can be used in other application contexts.

## Installation
The project is developed and tested with **Python 3.11**. It is recommended to use
a virtual environment derived from this version of Python to install the package.

1. Clone the repository
```bash
git clone https://gitlab.dlr.de/w2/w2-implementation.git
cd w2-implementation
```

2. Create a virtual environment
```bash
python -m venv w2-venv
```

2. Activate the virtual environment
```bash
source w2-venv/bin/activate
```

3. Install the package and requirements
```bash
pip install -r requirements.txt
```


## Repository structure
This repository contains the code and test data for the project. 
The code is organized as follows:

- `w2/`: contains the source code for the project with the main modules being:
  - `uncertainty/`: contains the modules for uncertainty estimation
  - `utils/`: contains the utility modules
  - `xai/`: contains the modules for eXplainable AI

- `test_data/`: contains the data used in the project
- `scripts/`: contains demo scripts to run the code on example data
- `use_cases/`: contains the use-case applications for the project


## Getting started: Post-hoc Uncertainty Quantification Approaches for Flood Detection from SAR Imagery
To run initial tests you can try the following demo script.

- `python use-cases/flood-detection/flood_uncertainty.py`: runs the uncertainty estimation on SAR flood example data


## Getting started: Other use-cases
To get started with the project, you can run the demo scripts in the `scripts/` folder.

- `python scripts/w2_uc_demo.py`: runs the uncertainty estimation on arbitrary example data
- `python scripts/w2_xai_demo.py`: runs the eXplainable AI on example data


## License
The project is still under development of DLR-PF/HR/WF/BT and not yet licensed.
The code should be treated as confidential and should not be shared with third parties.