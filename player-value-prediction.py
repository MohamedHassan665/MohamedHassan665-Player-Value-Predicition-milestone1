from tkinter import font
from turtle import width
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import *
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
import pickle
def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X
def featureScaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i]-min(X[:, i])) /
                              (max(X[:, i])-min(X[:, i])))*(b-a)+a
    return Normalized_X
data = pd.read_csv('player-value-prediction.csv')
# View Data
#print(data.describe())
#print(data.info())
#print(data.head())
#print("The number of rows in the dataset are:", data.shape[0])
#print("The number of columns in the dataset are:", data.shape[1])
##################################################################

# encoding for string values

data = data.drop(['id', 'full_name', 'birth_date'], axis=1)
# Drop Columns That Have Null Values >= 25%
for col in data.columns:
    val = data[col].isnull().sum()/data[col].count()*100
    if(val >= 25):
        #print(col + "="+str(val))
        data = data.drop(col, axis=1)
###################################################################
data['value'] = data['value'].fillna(data['value'].mean())
# Split Positions In Splited Columns
new_player_position = data['positions'].str.get_dummies(
    sep=',').add_prefix('position')
#print(new_player_position.head())
data = pd.concat([data, new_player_position], axis=1)
encod_cols = ['positions', 'nationality', 'preferred_foot',
              'work_rate', 'body_type', 'club_team', 'club_position']
data = Feature_Encoder(data, encod_cols)

#print(data.head())
#print(data.info())
columns = ['LS', 'ST', 'RS',
           'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM',
           'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB',
           'RCB', 'RB']
#print(data[columns])
###################################################################
# Split Values In Columns (+2)
for col in columns:
    data[col] = data[col].str.split('+', n=1, expand=True)[0]

#print(data[columns].head())
###################################################################
# Fill Null Values By 0
data[columns] = data[columns].fillna(0)
data[columns] = data[columns].astype(int)
#print(data[columns].info())
###################################################################
# Fill Null Values By Mean
data['shot_power'] = data['shot_power'].fillna(data['shot_power'].median())
data['dribbling'] = data['dribbling'].fillna(data['dribbling'].median())
#print(data['value'].head())
data['wage'] = data['wage'].fillna(data['wage'].mean())
###################################################################

# Draw The Relation Between The Preferred_Foot And (overall_rating,wage,value)
counts_preferred_foot = data["preferred_foot"].value_counts()
counts_preferred_foot = counts_preferred_foot.reset_index()
counts_preferred_foot.columns = ["preferred_foot", "Count"]
#print(counts_preferred_foot)
#sns.catplot(y="Count",
#            x="preferred_foot", data=counts_preferred_foot,
#            palette="RdBu",  aspect=2, kind="bar")
#plt.title("Figure : Preffered Foot Count Plot", fontsize=20)

#sns.catplot(x="preferred_foot", y="overall_rating", data=data,
#            aspect=2, kind="bar")
#plt.title("Figure : \n\n Overall Rating vs Preffered Foot",
 #         fontsize=20)
#sns.catplot(x="preferred_foot", y="wage", data=data, aspect=2, kind="bar")
#plt.title("Figure : \n\n PreferredFoot vs Wage",
 #         fontsize=20)
#sns.catplot(x="preferred_foot", y="value", data=data, aspect=2, kind="bar")
#plt.title("Figure : \n\n PreferredFoot vs value",
 #         fontsize=20)

#plt.show()
###################################################################
# Draw The Relation Between The Nationality And (overall_rating,wage,value)
#counts_Nationality = data["nationality"].value_counts()
#counts_Nationality = counts_Nationality.reset_index()
#counts_Nationality.columns = ["Nations", "Counts"]
#print(counts_Nationality.head())
#sns.catplot(y="Nations",
 #           x="Counts", data=counts_Nationality.head(30),
  #          aspect=2, kind="bar")
#plt.title("Figure : \n\n\n Nation Wise Players Counts",
 #         fontsize=20)

#counts_Nationality_top20 = counts_Nationality.iloc[0:20, :]
#print(counts_Nationality_top20)

#avgwageoverall = data.groupby("nationality", as_index=False)[
 #   "wage", "overall_rating"].mean()
#avgwageoverall.sort_values(by="wage", inplace=True, ascending=False)
#print(avgwageoverall.head())
#top10bywage = avgwageoverall.iloc[0:10, :]
#sns.catplot(y="nationality",
 #           x="wage", data=top10bywage,
  #          palette="RdBu", height=6, kind="bar", aspect=2)
#plt.title("Figure : \n\n\n Top 10 Country with highest mean Wage",
 #         fontsize=20)

#avgwageoverall.sort_values(by="overall_rating", inplace=True, ascending=False)
#top10byoverall = avgwageoverall.iloc[0:10, :]
#sns.catplot(y="nationality",
 #           x="overall_rating", data=top10byoverall,
  #          height=6, kind="bar", aspect=2)
#plt.title("Figure : \n\n\n Top 10 Country with highest mean Overall Rating",
 #         fontsize=30)


#plt.show()
###################################################################
# Draw The Relation Between The Age And (overall_rating,wage,value)
#sns.lmplot(x="age", y="wage", data=data,
 #          order=2, ci=None, scatter_kws={"color": "blue"},
  #         line_kws={"linewidth": 3, "color": "red"}, aspect=2)
#plt.title("Figure : \n\n\n Age vs Wage")
#sns.lmplot(x="age", y="overall_rating", data=data, markers="*",
 #          order=2, ci=None, scatter_kws={"color": "blue"},
  #         line_kws={"linewidth": 3, "color": "red"}, aspect=2)
#plt.title("Figure : \n\n\n Age vs overall_rating")
#sns.lmplot(x="age", y="value", data=data, markers="*",
 #          order=2, ci=None, scatter_kws={"color": "blue"},
  #         line_kws={"linewidth": 3, "color": "red"}, aspect=2)
#plt.title("Figure : \n\n\n Age vs value")
#plt.show()

#print(data['age'].describe())
#data.loc[data['overall_rating'] == data['overall_rating'].max(
#)][['name', 'age', 'overall_rating', 'international_reputation(1-5)']]
#data.loc[data['overall_rating'] == data['overall_rating'].min()][[
#    'name', 'age', 'overall_rating']]
###################################################################
# Draw The Relation Between The Club_Team And (overall_rating,wage,value)
data['club_team'] = data['club_team'].fillna('NA')
#avgwageoverall = data.groupby("club_team", as_index=False)[
 #  "wage", "overall_rating", "value"].mean()
#avgwageoverall.sort_values(by="wage", inplace=True, ascending=False)
#print(avgwageoverall.head())

#top10bywage = avgwageoverall.iloc[0:10, :]
#sns.set(rc={"font.style": "normal",
 #           "axes.facecolor": (0.9, 0.9, 0.9),
  #          "figure.facecolor": 'white',
   #         'axes.labelsize': 30,
    #        'xtick.labelsize': 25,
     #       'ytick.labelsize': 20})
#sns.catplot(y="club_team",
 #           x="wage", data=top10bywage,
  #          height=6, kind="bar", aspect=2)
#plt.title("Figure : \n\n\n Top 10 Country with highest mean Wage")

#avgwageoverall.sort_values(by="overall_rating", inplace=True, ascending=False)
#top10byoverall = avgwageoverall.iloc[0:10, :]
#sns.catplot(y="club_team",
 #           x="overall_rating", data=top10byoverall,
  #          height=6, kind="bar", aspect=2)
#plt.title("Figure : \n\n\n Top 10 Country with highest mean Overall Rating")
#avgwageoverall.sort_values(by="value", inplace=True, ascending=False)
#top10byoverall = avgwageoverall.iloc[0:10, :]
#sns.catplot(y="club_team",
 #           x="value", data=top10byoverall,
  #          height=6, kind="bar", aspect=2)
#plt.title("Figure : \n\n\n Top 10 Country with highest mean value")
#plt.show()
###################################################################
# Draw The Relation Between The Height_Cm And (overall_rating,wage,value)
#dataHeight = data.loc[:, ["height_cm", "wage","overall_rating", "value"]].sort_values("height_cm")
#print(dataHeight.head())
#avgwageoverall = data.groupby("height_cm", as_index=False)[
 #   "wage", "overall_rating", "value"].mean()
#avgwageoverall.sort_values(by="wage", inplace=True, ascending=False)
#print(avgwageoverall.head())
#sns.set(rc={"font.style": "normal",
 #           "axes.facecolor": (0.9, 0.9, 0.9),
  #          "figure.facecolor": 'white',
   #         'axes.labelsize': 30,
    #        'xtick.labelsize': 5,
     #       'ytick.labelsize': 20})
#sns.catplot(x="height_cm", y="wage", data=avgwageoverall,
 #           kind="bar", aspect=2.5)
#plt.title("Figure : \n\n\n Height vs wage")
#avgwageoverall.sort_values(by="value", inplace=True, ascending=False)
#sns.catplot(x="height_cm", y="value", data=avgwageoverall,
 #           kind="bar", aspect=2.5)
#plt.title("Figure : \n\n\n Height vs value")
#avgwageoverall.sort_values(by="overall_rating", inplace=True, ascending=False)
#sns.catplot(x="height_cm", y="overall_rating", data=avgwageoverall,
 #           kind="bar", aspect=2.5)
#plt.title("Figure : \n\n\n Height vs Overall_rating")
#plt.show()
###################################################################
# Draw The Relation Between The Weight_kgs And (overall_rating,wage,value)
#dataHeight = data.loc[:, ["weight_kgs", "wage","overall_rating", "value"]].sort_values("weight_kgs")
#print(dataHeight.head())
#avgwageoverall = data.groupby("weight_kgs", as_index=False)[
 #   "wage", "overall_rating", "value"].mean()
#avgwageoverall.sort_values(by="wage", inplace=True, ascending=False)
#print(avgwageoverall.head())
#sns.set(rc={"font.style": "normal",
 #           "axes.facecolor": (0.9, 0.9, 0.9),
  #          "figure.facecolor": 'white',
   #         'axes.labelsize': 30,
    #        'xtick.labelsize': 5,
     #       'ytick.labelsize': 20})
#sns.catplot(x="weight_kgs", y="wage", data=avgwageoverall,
 #           kind="bar", aspect=2.5)
#plt.title("Figure : \n\n\n Weight vs wage")
#avgwageoverall.sort_values(by="value", inplace=True, ascending=False)
#sns.catplot(x="weight_kgs", y="value", data=avgwageoverall,
 #           kind="bar", aspect=2.5)
#plt.title("Figure : \n\n\n Weight vs Value")
#avgwageoverall.sort_values(by="overall_rating", inplace=True, ascending=False)
#sns.catplot(x="weight_kgs", y="overall_rating", data=avgwageoverall,  kind="bar", aspect=2.5)
#plt.title("Figure : \n\n\n Weight vs Overall_rating")
#plt.show()
#sns.lmplot(x="potential", y="overall_rating", data=data, markers="*",           order=2, ci=None, scatter_kws={"color": "darkgreen"},           line_kws={"linewidth": 3, "color": "red"}, aspect=2)
#plt.title("Figure : \n\n\n Overall Rating vs potential")
#sns.lmplot(x="potential", y="value", data=data, markers="*",
 #          order=2, ci=None, scatter_kws={"color": "darkgreen"},
  #         line_kws={"linewidth": 3, "color": "red"}, aspect=2)
#plt.title("Figure : \n\n\n value vs potential")
#sns.lmplot(x="potential", y="wage", data=data, markers="*",
 #          order=2, ci=None, scatter_kws={"color": "darkgreen"},
  #         line_kws={"linewidth": 3, "color": "red"}, aspect=2)
#plt.title("Figure : \n\n\n Wage vs potential")
#plt.show()
data.dropna(how='any', inplace=True)
###################################################################
corr = data.corr()
# Top 40% Correlation training features with the Value
top_feature = corr.index[abs(corr['value']) > 0.35]
# Correlation plot
#plt.subplots(figsize=(12, 8))
#top_corr = data[top_feature].corr()
#sns.heatmap(top_corr, annot=True)
#plt.show()
plotting_columns = ['reactions', 'club_rating', 'release_clause_euro',
                    'international_reputation(1-5)', 'wage', 'potential', 'overall_rating']

data[plotting_columns] = data[plotting_columns].fillna(0)
#data['value'] = data['value'].fillna(data['value'].mean())

####################################################################################
##################           Regression Part           ####################
top_feature = top_feature.delete(-1)
top_feature = top_feature.drop(['reactions', 'composure'])
#print(top_feature)
X = featureScaling(data[top_feature], 0, 1)
scaleValues = MinMaxScaler()
# scaleValues.fit_transform(data[['value']])

X_train, X_test, y_train, y_test = train_test_split(X, data['value'], test_size=0.20, shuffle=True, random_state=10)


start_time = time.time()
M_L_R = linear_model.LinearRegression()
M_L_R.fit(X_train, y_train)
start_time = time.time()
prediction = M_L_R.predict(X_test)

print("Mean Square Error at Linear Regression :",
      metrics.mean_squared_error(np.asarray(y_test), prediction))
print("--- %s seconds ---" % (time.time() - start_time))
print('accuracy = ', M_L_R.score(X_test, y_test)*100)


model_1_poly_features = PolynomialFeatures(degree=3)
X_train_poly_model_1 = model_1_poly_features.fit_transform(X_train)
# fit the transformed features to Linear Regression
poly_model1 = linear_model.LinearRegression()
scores = cross_val_score(poly_model1, X_train_poly_model_1,y_train, scoring='neg_mean_squared_error', cv=6)
model_1_score = abs(scores.mean())
poly_model1.fit(X_train_poly_model_1, y_train)
prediction = poly_model1.predict(model_1_poly_features.fit_transform(X_test))
print('Mean Square Error with cross validation : ',
      metrics.mean_squared_error(y_test, prediction))
print("--- %s seconds ---" % (time.time() - start_time))

with open('PolynomialRegression model with cross validation','wb')as f:
    pickle.dump(model_1_poly_features,f)

start_time = time.time()
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_train_predicted = poly_model.predict(X_train_poly)
ypred = poly_model.predict(poly_features.transform(X_test))
prediction = poly_model.predict(poly_features.fit_transform(X_test))
print('Mean Square Error without cross validation : ',
      metrics.mean_squared_error(y_test, prediction))
print("--- %s seconds ---" % (time.time() - start_time))


with open('PolynomialRegression model without cross validation','wb')as f:
    pickle.dump(poly_features,f)





ridge_model=Ridge(alpha = 1).fit(X_train, y_train)
ridge_model = Ridge().fit(X_train, y_train)
y_pred = ridge_model.predict(X_train)
RMSE = (mean_squared_error(y_train, y_pred))
print('Mean Square Error of Ridge Model ',RMSE)

with open('Ridge model','wb')as f:
    pickle.dump(ridge_model,f)


