#### Homework 3 by Group 9 (Emily Schoener, Josh Velazquez, and John Ingram)
## Requirements if you want to run the code without using Conda:
```
python 3.11 (although any version of python 3 should work)
numpy
scikit-learn
scipy
```

How to run the code:

Make sure `Homework3.py` and `iris.txt` are in the same directory.

**FROM THE DIRECTORY CONTAINING THE FILES:** 
run `python Homework3.py` in the terminal that has python in its path (this should be powershell on windows and terminal on mac/linux)

# IF YOU ARE HAVING TROUBLE RUNNING THE CODE, PLEASE USE THE CONDA ENVIRONMENT:
- Extract the Group9HW3.zip file to a convenient location
- Install and update Miniconda if you haven't already
    - Go [here](https://docs.conda.io/en/latest/miniconda.html) to install the correct Miniconda version for your system **Make sure you acccept the default settings**
- Open the Anaconda Prompt (Windows) or Terminal (Mac/Linux)
- Navigate to the the Group9HW2 folder
    - To do this, use the command `cd <path to Group9HW2>`
- To create the environment, run
```
conda create --name Group9HW3 -c anaconda numpy scikit-learn scipy 
``` 
to create the environment
- Run `conda activate Group9HW3` to activate the environment
- Run `python Homework3.py` to run the code

