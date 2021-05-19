# Gapminder Data

# **Data**

The Gapminder data which was analysed can be accessed here [https://www.gapminder.org/data/](https://www.gapminder.org/data/) Data was gathered for life expectancy, income, population and HDI (human development index) which looks at factors such as education and opportunity. The motivation behind the project was to assess the effect of different factors on life expectancy.

# Questions

### Question 1 - What's the effect of income on life expectancy. Or does money buy you more time on Earth?

This question is posed by most people. Will that new job, or that raise at work buy me happiness? We wanted to explore that in more detail based on socioeconomic data available for different countries. The variables gathered were income, population and the human development index of each country which is defined as the level of education and opportunity available to that individual in his or her respective country.

The approach to be taken is through visualising the data and telling a story with the data. We went with an interactive html visualisation using the plotly library to visualise the change in life expectancy over time as income increased in various countries sized by their population.

```python
class Visualisation():
    def __init__(self):
        self.df = pd.DataFrame()
    
    def exploration(self,plot,x=None,y=None):
        sns.set_style("darkgrid")
        if plot == "regression":
            sns.regplot(x=x,y=y,data=self)
            plt.show()
        elif plot == "heatmap":
            sns.heatmap(self.corr())
            plt.show()

    def animation(self,col,save):
        fig = px.scatter(self, x=col, y="life_expectancy", animation_frame="year", animation_group="country",
           size="population", color="country", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])
        if save == True:
            fig.write_html(f"./visualisations/{col}_vs_life_expectancy.html")
	    fig.show()
        else:
            fig.show()

```

### Question 2 - How do the independent variables affect our dependent variable?

#### Uni-variate analysis
Population variable was visualised in a box plot. Results showed large amount of outliers which indicates there were issues when collecting the data
and issues of data quality.

#### Multi-variate analysis
Correlation was visualised for each plot against its pair using pair plots for understanding the effect of each vairable on the other.

We wanted to explore the effect of our variables on life expectancy. This was carried out in the form of correlation analysis to see how they interacted with each other. The dataset was encoded and split into a training and testing data to see if we could predict the life expectancy of a small sample of the dataset (33%) and based on the small number of features used a model accuracy of almost 80% was achieved using linear regression methods.

Regression coefficients represent the mean change in the response variable for one unit of change in the predictor variable while holding other predictors in the model constant.

We could use more features either through engineering or bringing in more data via the ETL pipeline we have built for this project. It could also be interesting to run a KMeans Clustering algorithm on the dataset to group countries into separate categories of life expectancy metrics.

```python
class MachineLearning():
    def __init__(self):
        self.df = pd.DataFrame()

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
        print(f"""Accuracy score for a Bayseian model is {reg.score(X_test, y_test):.5f},
	          for a Linear model is {reg2.score(X_test, y_test):.5f}""")
```

# Data Wrangling

The data available homogeneously throughout the csv files was between 1990 and 2018. Some datasets had missing values whilst others did not and the datasets needed to be transposed and stacked in order to shift their presentation so that they could be all merged into one dataframe that we can carry out the analysis in.

To create a replicable program that could be altered to add further features as analysis advanced it needed to be carried out using Object Oriented Programming. Creating classes that can store code that carries out different functions on the data as it comes in the pipeline.

Summary statistics are printed out to the terminal on each dataset as well as on the data as a whole. Then the data is cleaned and merged after the transform process.

```python
class Gapminder():
    def __init__(self,filename):
        self.df = pd.read_csv(filename)
        self.filename = filename.split("/")[-1].strip(".csv")
    
    def summary(self):
        print(self.columns.tolist())
        print("Shape of the dataframe is ",self.shape)
        print(self.isnull().sum())
        print("Maximum date in data is {}\n Minimum date is {}".format(self.year.max(),self.year.min()))
        print("---")
        if len(self.columns.tolist()) >= 4:
            print(self.describe())

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

    def merge(self,*args):
        args = list(args)
        self.df = reduce(lambda l,r: pd.merge(l,r,on=["year","country"]), args)
        self.df.columns = ["year","country","population","life_expectancy","income","hdi"]
        return self.df

```

# Summary Statistics

### Dataset description of variables.

![Gapminder%20Data%20027cbe08f44c4a068acece11973d58f9/Untitled.png](Gapminder%20Data%20027cbe08f44c4a068acece11973d58f9/Untitled.png)

### Regression analysis

![Gapminder%20Data%20027cbe08f44c4a068acece11973d58f9/Untitled%201.png](Gapminder%20Data%20027cbe08f44c4a068acece11973d58f9/Untitled%201.png)

Improvements to the model would come in the form of more features in order for the algoriothm to build a better understanding on the effect of different factors on life expectancy.

# Plots

### Income vs Life Expectancy Scatter Plot

[![Scatter Plot](https://img.youtube.com/vi/fafzJoBXtyE/0.jpg)](https://youtu.be/fafzJoBXtyE)

[https://youtu.be/fafzJoBXtyE](https://youtu.be/fafzJoBXtyE)

Available as an interactive html file inside the visualisations folder

Income vs life expectancy for countries over the year span from 1990 to 2018 from Gapminder data.

This visualisation answers question 1, does money buy a longer life? This was the behaviour from 1990 to 2018 on the countries available in our analysis. A scatter plot was used with bubbles sized by population and coloured by country. Income seemed to be most correlated with life expectancy in the analysis so it made sense to visualise that as shown above and explained by the below visualisation of a regression plot between the two variables showing a strong positive correlation. 

### Regression Plot for Income vs Life Expectancy

![Gapminder%20Data%20027cbe08f44c4a068acece11973d58f9/Untitled%202.png](Gapminder%20Data%20027cbe08f44c4a068acece11973d58f9/Untitled%202.png)

### Heat Map for Correlation of Variables

![Gapminder%20Data%20027cbe08f44c4a068acece11973d58f9/Untitled%203.png](Gapminder%20Data%20027cbe08f44c4a068acece11973d58f9/Untitled%203.png)

Life expectancy is correlated with Human Development Index and Income.
