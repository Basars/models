# Basars Datasets

You can request dataset from [here](https://aihub.or.kr/aidata/33988).

## Recommended Dataset Hierarchy
```
Dataset:
- datasets
    - train
    - train_masks
    - test
    - test_masks
    - valid
    - valid_masks
    - train_labels.csv
    - test_labels.csv
    - valid_labels.csv
```

## Raw Dataset Hierarchy
```
Dataset:
- {LABEL_DIR}
    - *
        - ENDO
            - *.json
- {DCM_DIR}
    - *
        - ENDO
            - *.dcm
```

## Preprocessing

You must preprocess the raw datasets by hand first.

Once you've done preprocessing manually, You must create datasets using the [preprocessor](https://github.com/Basars/preprocessors).

