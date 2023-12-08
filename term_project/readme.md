#### Homework 4 by Group 9 (Emily Schoener, Josh Velazquez, and John Ingram)
## Requirements if you want to run Homework4 without using Conda:
```
python 3.10.13 (although any version of python 3 should work)
numpy
scikit-learn
scipy
matplotlib
tensorflow
keras
```

How to run the code:

Make sure `main.py` and `forestfires.csv` are in the same directory.

**FROM THE DIRECTORY CONTAINING THE FILES:** 
run `python main.py` in the terminal that has python in its path (this should be powershell on windows and terminal on mac/linux)

# To simplify running the code, PLEASE USE THE CONDA ENVIRONMENT:
- Extract the Group9FinalProj.zip file to a convenient location
- Install and update Miniconda if you haven't already
    - Go [here](https://docs.conda.io/en/latest/miniconda.html) to install the correct Miniconda version for your system **Make sure you acccept the default settings**
- Open the Anaconda Prompt (Windows) or Terminal (Mac/Linux)
- Navigate to the the Group9FinalProj folder
    - To do this, use the command `cd <path to Group9FinalProj>`
- To create the environment, run
```
conda create --name Group9FinalProj -c anaconda numpy scikit-learn scipy matplotlib tensorflow keras
``` 
to create the environment
- Run `conda activate Group9FinalProj` to activate the environment
- Run `python main.py` to run the code
- to run the code without generating the plots, run `python main.py -ng`
