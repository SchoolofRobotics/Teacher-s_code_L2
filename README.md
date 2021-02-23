# Teacher-s_code_L2

This is a repository for Machine Learning agents in Game Design lesson 2. Teacher code.

## Training

To run Evolutionary Algorithm to train, type in command line this command:

```
python evolution.py
```

## Inference

After training, top individual parameters will be saved in `results` folder. To test single individual, run:

```
python evo_inference.py <path_to_individual>
```

#### For example:
`python evo_inference.py results\individual1.txt`
