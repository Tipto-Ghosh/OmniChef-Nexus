* we want to use this dataset
    - Dataset: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions


* we have total 231637 samples. For the first trial we will limit our self to 20 ingredients and 20 steps.
* So we want around ~15k samples
  - we will filter our dataset based on n_steps <= 20 and n_ingredients <= 20
  - Then we will take randomly 15k samples from the subset.

  