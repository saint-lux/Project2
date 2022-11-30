Step 1: Create conda environment with python 3.6
Environment Name: audioml

```
conda create -n "audioml" python=3.6
```

Enter the environment and check python version
```
conda activate audioml

python -V
```


Step 2: Install pyaudio and scikit learn with dependencies

```
conda install -c anaconda scikit-learn

conda install -c anaconda pyaudio
```
Note: this will ensure that we have all the dependencies needed to record audio, train and test the ML model.

Step 3: Install Jupyter Lab in this environment
```
conda install -c conda-forge jupyterlab
```