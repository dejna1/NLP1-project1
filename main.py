from datasets import load_dataset, DatasetDict, concatenate_datasets


def retrieve_datasets() -> tuple:
    # load ds from HF
    ds = load_dataset("cardiffnlp/tweet_eval", "offensive")

    # train: 11916, validation: 1324, test:860
    X_train = ds['train']['text']
    y_train = ds['train']['label']
    X_validation = ds['validation']['text']
    y_validation = ds['validation']['label']
    X_test = ds['test']['text']
    y_test = ds['test']['label']
    
    # consider returning only train, test splits - save validation
    # for hyperparameter tuning
    
    return X_train, y_train, X_validation, y_validation, X_test, y_test

# tokenize text 
# train - k-cross-validation
# evaluate
# save preds
# metrics
# compare with bayesian ROPE