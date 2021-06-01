# Importing Libraries
import numpy as np
import pandas as pd


class GaussianNB():
    def __init__(self):
        """
        Has fit() method and predict() method. 
        """
        self.mean = None
        self.variance = None
        self.target = None
        self.target_class = None
        self.target_class_count = None

    def _get_main_proba(self):
        return [x/sum(self.target_class_count) for x in self.target_class_count]

    def fit(self, train_x, train_y):
        """
        Fit Gaussian Naive Bayes Model

        Parameters
        ----------
            train_x: DataFrame
            train_y: DataFrame
        """
        # Preprocess
        df = train_x
        df["target"] = train_y

        # Mean and Variance
        grp_target = df.groupby(['target'])
        self.target_class_count = grp_target.count().iloc[:, 1].to_list()
        self.target_class = grp_target.mean().index.tolist()
        self.mean = grp_target.mean().to_numpy()
        self.variance = grp_target.var().to_numpy()

    def predict(self, test_x):
        """
        Predict Class with Confidence

        Parameters
        ----------
            test_x: List
        
        Returns
        -------
            Dictionary
        """
        x = np.array([test_x, ]*len(self.target_class))
        overall_probability = np.exp(-(np.square(x-self.mean)) /
                                     (2*self.variance)) / np.sqrt(2*3.14*self.variance)

        proba = np.prod(overall_probability, axis=1)
        fin_prob = list(proba * self._get_main_proba())
        return {'predicted': self.target_class[fin_prob.index(max(fin_prob))],
                'confidence': max(fin_prob)}


if __name__ == '__main__':
    # Instantiate GaussianNB class
    nb = GaussianNB()
    train_data = pd.read_csv('gender_data.csv')

    # Loading dataset
    train_x = train_data.drop(columns=["Person"], axis=1)
    train_y = train_data["Person"]

    # Training dataset
    nb.fit(train_x, train_y)

    # Predict
    prediction = nb.predict([6, 130, 8])

    # Display Prediction
    print(prediction)
