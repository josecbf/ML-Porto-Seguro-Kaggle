# ML-Porto-Seguro-Kaggle

# Porto Seguro’s Safe Driver Prediction

### Chalenge Description

Nothing ruins the thrill of buying a brand new car more quickly than seeing your new insurance bill. The sting’s even more painful when you know you’re a good driver. It doesn’t seem fair that you have to pay so much if you’ve been cautious on the road for years.

Porto Seguro, one of Brazil’s largest auto and homeowner insurance companies, completely agrees. Inaccuracies in car insurance company’s claim predictions raise the cost of insurance for good drivers and reduce the price for bad ones.

In this competition, you’re challenged to build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year. While Porto Seguro has used machine learning for the past 20 years, they’re looking to Kaggle’s machine learning community to explore new, more powerful methods. A more accurate prediction will allow them to further tailor their prices, and hopefully make auto insurance coverage more accessible to more drivers.

#### Link: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction

### Evaluation

Submissions are evaluated using the Normalized Gini Coefficient.

During scoring, observations are sorted from the largest to the smallest predictions. Predictions are only used for ordering observations; therefore, the relative magnitude of the predictions are not used during scoring. The scoring algorithm then compares the cumulative proportion of positive class observations to a theoretical uniform proportion.

The Gini Coefficient ranges from approximately 0 for random guessing, to approximately 0.5 for a perfect score. The theoretical maximum for the discrete calculation is (1 - frac_pos) / 2.

The Normalized Gini Coefficient adjusts the score by the theoretical maximum so that the maximum score is 1.


### Instalation

This project requires Python 3.6

And the following libraries (should work in all versions)

- pandas
- numpy
- time
- sklearn
- xgboost
- seaborn
- matplotlib.pyplot
- missingno


### Execution

To run this project and get the same submission file you have to dowload the data in the link below and put the train.csv and test.csv in the folder udacity-data

Then you have run this command: python3 final_model.py

The submission file will be in the folder output
