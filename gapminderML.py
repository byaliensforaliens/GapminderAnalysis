import numpy as np
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model

class Gapminder():
    # instantiating the class to carry out data wrangling
    def __init__(self,filename):
        self.df = pd.read_csv(filename)
        self.filename = filename.split("/")[-1].strip(".csv")
    
    # function was used initially to learn about the data and plan preprocessing
    def summary(self):
        print(self.columns.tolist())
        print("Shape of the dataframe is ",self.shape)
        print(self.isnull().sum())
        print("Maximum date in data is {}\n Minimum date is {}".format(self.year.max(),self.year.min()))
        print("---")
        if len(self.columns.tolist()) >= 4:
            print(self.describe())

    # this function carries out the heavy lifting shifting things around, dealing with missing values and changing data types
    def clean(self,imputer):
        self.df = self.df.T
        self.df.columns = self.df.iloc[0]
        self.df = self.df.iloc[1:,:]
        if imputer == False:
            self.df.dropna(axis=1,inplace=True)
        elif imputer == True:
            for i in self.df.columns.tolist():
                self.df[i].fillna((self.df[i].mean()), inplace=True)
        self.df = self.df.stack(0)
        self.df.index.set_names('year', level=len(self.df.index.names)-2,inplace=True)
        self.df = self.df.reset_index().rename(columns={0:f'{self.filename}'})
        self.df = self.df[(self.df['year'] >= '1990-01-01') & (self.df['year'] <= '2018-01-01')]
        num_cols = self.df.columns[-1:].tolist()
        for i in num_cols:
            self.df[i] = pd.to_numeric(self.df[i])
        return self.df

    # this function merges the individual dataframes into one dataset that can be used for analysis, visualisation and machine learning
    def merge(self,*args):
        args = list(args)
        self.df = reduce(lambda l,r: pd.merge(l,r,on=["year","country"]), args)
        self.df.columns = ["year","country","population","life_expectancy","income","hdi"]
        return self.df

class Visualisation():
    # instantiating the class to carry out EDA visualisation
    def __init__(self):
        self.df = pd.DataFrame()
    
    # function uses seaborn to plot correlation heatmap to explore variable relationship and regression plot for columns 
    def exploration(self,plot,x=None,y=None):
        sns.set_style("darkgrid")
        if plot == "regression":
            sns.regplot(x=x,y=y,data=self)
            plt.show()
        elif plot == "heatmap":
            sns.heatmap(self.corr())
            plt.show()

    # using an interactive scatter plot from the plotly library we can observe the behaviour of variables against life expectancy over time
    def animation(self,col,save):
        fig = px.scatter(self, x=col, y="life_expectancy", animation_frame="year", animation_group="country",
           size="population", color="country", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
        if save == True:
            fig.write_html(f"./visualisations/{col}_vs_life_expectancy.html")
            fig.show()
        else:
            fig.show()

class MachineLearning():
    # instantiating machine learning class to carry out predictions on life expectancy based on independent variables
    def __init__(self):
        self.df = pd.DataFrame()

    # function encodes the country column and gives it dummy variables to be processed by the regression algorithm
    def encoding(self):
        self = self[["year","country","population","income","hdi","life_expectancy"]]
        self["year"] = pd.to_datetime(self["year"])
        self["year"] = self["year"].dt.year
        cols = list(self.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in cols:
            try:
                self[feature] = le.fit_transform(self[feature])
            except:
                print('Error encoding '+feature)
        return self
   
    # splits the dataset into training and testing sets and stacks together two linear regression models to generate coefficients and accuracy scores
    def run(self):
        X = self[self.columns[:-1]]
        y = self[self.columns[-1]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        reg = linear_model.BayesianRidge()
        reg.fit(X_train,y_train)
        y_pred = reg.predict(X_test)
        reg2 = linear_model.LinearRegression()
        reg2.fit(X_train,y_train)
        y_pred2 = reg2.predict(X_test)
        cdf = pd.DataFrame(reg.coef_, X.columns, columns=['Coefficients'])
        print(f"""{cdf}\n
        Regression coefficients represent the mean change in the response variable
        for one unit of change in the predictor variable
        while holding other predictors in the model constant.\n""")
        print(f"Accuracy score for a Bayseian model is {reg.score(X_test, y_test):.5f}, for a Linear model is {reg2.score(X_test, y_test):.5f}")

# bulk of work carried out in this function to bring everything together
def main():
    population = Gapminder("./gapminder_data/population_total.csv").clean(imputer=False)
    life_expectancy = Gapminder("./gapminder_data/life_expectancy_years.csv").clean(imputer=False)
    income = Gapminder("./gapminder_data/income_per_person_gdppercapita_ppp_inflation_adjusted.csv").clean(imputer=False)
    human_development = Gapminder("./gapminder_data/hdi_human_development_index.csv").clean(imputer=True)
    data = pd.DataFrame()
    data = Gapminder.merge(data,population,life_expectancy, income, human_development)
    Gapminder.summary(population)
    Gapminder.summary(life_expectancy)
    Gapminder.summary(income)
    Gapminder.summary(human_development)
    Gapminder.summary(data)   
    Visualisation.exploration(data,plot="heatmap")
    Visualisation.exploration(data,plot="regression",x="income",y="life_expectancy")
    Visualisation.animation(data,"income",save=False)
    Visualisation.animation(data,"hdi",save=True)
    df = MachineLearning.encoding(data)
    MachineLearning.run(df)
    
# runs program
if __name__ == '__main__':
    main()