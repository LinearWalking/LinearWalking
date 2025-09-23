This repository contains an example code notebook for our paper "The Surprising Effectiveness of Linear Models for Whole-Body Model-Predictive Control". Find the project website [here](https://linearwalking.github.io/).

# Installation
### 1. Clone the repository
```
https://github.com/LinearWalking/LinearWalking
cd LinearWalking
```

### 2. Create and Activate a Virtual Environment (optional but recommended)
```
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
pip install numpy matplotlib h5py osqp scipy mediapy notebook

### 4. Run the example notebook `linear_walking_quadruped.ipynb`
The problem data for the QP was generated in Julia. For more details, check out our [website](https://linearwalking.github.io/) and paper.