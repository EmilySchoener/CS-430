## Requirements if you want to run the code without using Conda:
```
python 3.11 (although any version of python 3 should work)
matplotlib
```

How to run the code:

Make sure `JohnIngramHW1.py` and `data.txt` are in the same directory.

**FROM THE DIRECTORY CONTAINING THE FILES:** 
run `python JohnIngramHW1.py` in the terminal that has python in its path (this should be powershell on windows and terminal on mac/linux)

# IF YOU ARE HAVING TROUBLE RUNNING THE CODE, PLEASE USE THE CONDA ENVIRONMENT:
- Extract the JSIHomework1.zip file to a convenient location
- Install and update Miniconda if you haven't already
    - Go [here](https://docs.conda.io/en/latest/miniconda.html) to install the correct Miniconda version for your system **Make sure you acccept the default settings**
- Open the Anaconda Prompt (Windows) or Terminal (Mac/Linux)
- Navigate to the the JSIHomework1 folder
    - To do this, use the command `cd <path to JSIHomework1>`
- Run `conda env create -f JohnIngramHW1env.yml` to create the environment
- Run `conda activate JohnIngramHW1env` to activate the environment
- Run `python JohnIngramHW1.py` to run the code

