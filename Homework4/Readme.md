#### Homework 4 by Group 9 (Emily Schoener, Josh Velazquez, and John Ingram)
## Requirements if you want to run Homework4 without using Conda:
```
python 3.11 (although any version of python 3 should work)
numpy
scikit-learn
scipy
matplotlib
libsvm
```

How to run the code:

Make sure `Homework4.py` and `iris.txt` are in the same directory.

**FROM THE DIRECTORY CONTAINING THE FILES:** 
run `python Homework4.py` in the terminal that has python in its path (this should be powershell on windows and terminal on mac/linux)

# IF YOU ARE HAVING TROUBLE RUNNING THE CODE, PLEASE USE THE CONDA ENVIRONMENT:
- Extract the Group9HW4.zip file to a convenient location
- Install and update Miniconda if you haven't already
    - Go [here](https://docs.conda.io/en/latest/miniconda.html) to install the correct Miniconda version for your system **Make sure you acccept the default settings**
- Open the Anaconda Prompt (Windows) or Terminal (Mac/Linux)
- Navigate to the the Group9HW4 folder
    - To do this, use the command `cd <path to Group9HW4>`
- To create the environment, run
```
conda create --name Group9HW4 -c anaconda numpy scikit-learn scipy matplotlib
``` 
to create the environment
- Run `conda activate Group9HW4` to activate the environment
- Run `pip install libsvm` to install the libsvm package, since this package cannot be reliably installed by conda itself
- Run `python Homework4.py` to run the code
