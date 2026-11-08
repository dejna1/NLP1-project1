# Natural Language Processing 1 - Project 1

## Authors: Daniella Alves, Lam Anh Nguyen, Söhret Öznacar

### Repository skeleton
```
NLP-project1/
├── models/
│   ├── logreg.py
│   ├── mlp.py      # file entirely taken from the tutorial
│   ├── random_forest.py    # chosen additional model
├── results/
│   ├── models/     # saved models
│   ├── metrics.csv     # exahustive summary of 9 system combinations x 5 fold cross-validation x 3 seeds metric performance
│   ├── rope_comparisons.csv       # results from comparing pairs of systems using Bayesian ROPE analysis
│   ├── summary.csv    # F1-score & Accuracy results over the averaged 9 systems
│   ├── tokenizer.json      # built vocabulary 
├── text_reps/
│   ├── representations.py     # 3 representations spaces
│   ├── tokenization.py     # main functions used to manipulate tokenizers
├── analysis.py     # functions to compute Bayesian ROPE analysis
├── evaluation.py      # contains functions to run main 9 system combinations x 5 fold cross-validation x 3 seeds experiment
├── main.py
├── README.md
├── tut_rope.py     # file entirely taken from the tutorial
```

### AI-use disclosure
Generative AI tools were used to create code that helps with training progress tracking, results saving (paths, configurations, csv files), and polishing.