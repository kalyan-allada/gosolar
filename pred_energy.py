import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score
from sklearn import model_selection
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score, ShuffleSplit
from sklearn.model_selection import learning_curve, train_test_split
from sklearn import preprocessing
import seaborn as sns
import pickle

def read_data():
    data = pd.read_csv("data/openpv_ma_final.csv", dtype={'zipcode': 'str'})
    return data

def encode_features(data):
    # Label encode "tech_1" and "tracking_type"           
    tech_dict = {'Poly': 1, 'Mono': 2, 'crystalline': 3, 'Mono + a-Si':4, 'a-Si':5,\
                 'multiple':6}
    track_dict = {'Fixed': 1, 'Single-Axis': 2, 'Dual-Axis': 3}
    data['tech_1'] = data['tech_1'].map(tech_dict).astype(int)
    data['tracking_type'] = data['tracking_type'].map(track_dict).astype(int)
    return data
    
def cross_validate(data):
    features_to_use = ["size_kw", "annual_insolation", "tilt1", "tracking_type",\
                       "tech_1" , "Lat", "Long"]
    
    X = data[features_to_use].values
    y = data["reported_annual_energy_prod"].values
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, \
                                                    random_state=42)
    #regressor = LinearRegression()
    #regressor = DecisionTreeRegressor()
    
    # Hyperparameters optimized using gridsearch cv
    regressor = RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_leaf=5)
    
    cv_ssplit = ShuffleSplit(n_splits=10, test_size=0.3, random_state=100)
    cv_score = cross_val_score(regressor, X_train, y_train, cv=cv_ssplit, scoring = 'r2')
    
    print("Average accuracy: %0.4f +/- %0.4f" % (cv_score.mean(), cv_score.std()*2))
    return (regressor, X_train, X_test, y_train, y_test)

def do_grid_search(regressor, X_train, y_train): 
    params = {
              "n_estimators":[50, 100, 150],
              "min_samples_split":[2, 5, 10],
              "min_samples_leaf":[5, 10, 50],
              #"max_features": ['auto', 'sqrt'],
              "max_depth":[20, 50, 90]
                 }
    cv_reg = GridSearchCV(regressor, param_grid=params, cv=3)
    cv_reg.fit(X_train, y_train)
    print cv_reg.best_params_
    print cv_reg.best_estimator_
    print cv_reg.best_score_
    return

def make_predictions(regressor, X_train, X_test, y_train):
    #Fit training data 
    regressor = regressor.fit(X_train, y_train)
    y_valid = regressor.predict(X_train)
    
    #Make predictions on test set
    y_pred = regressor.predict(X_test)
    return (y_pred, y_valid)

def dump_model(regressor):   
    with open('data/energy_ma_final.pk','wb') as f:
        pickle.dump(regressor, f)
    return

def compute_error(y_test, y_pred):    
    
    print("Mean abs error: %.2f" % mean_absolute_error(y_test, y_pred))
    print('Variance score: %.2f' % r2_score(y_test, y_pred))  
    print('RMSE %.2f' % mean_squared_error(y_test, y_pred)**0.5)
      
    return
    
def plot(regressor, y_train, y_valid, y_test, y_pred):
    residuals_train =  y_valid - y_train
    residuals_test =  y_pred - y_test
    
    fig1 = plt.figure(figsize=(14,12)) 
    sns.regplot(np.array(y_valid), residuals_train, color='g', marker='o', \
                scatter_kws={'alpha':0.3})
    sns.regplot(np.array(y_pred), residuals_test, color='r', marker='o', \
                scatter_kws={'alpha':0.3})

    plt.title("Residuals for training (green) and test (red) data", fontsize=40)
    plt.ylabel("Residuals",fontsize=40)
    plt.xlabel("Reported energy production (kWh/year)",fontsize=40)
    plt.tick_params(labelsize=30)
    plt.xlim(0,0.5e5)
    plt.ylim(-1.5e4,1.5e4)
    #fig1.savefig('./plots/residuals_lr.png')
    
    fig2 = plt.figure(figsize=(12,10)) 

    features = ["Array size", "Annual Insolation", "Tilt angle", "Tracking type", \
                "Array type", "Latitude", "Longitude"]
    
    feat_imp_df = pd.DataFrame({"feature":features, "importance":regressor.feature_importances_})
    feat_imp_df.sort_values("importance", inplace=True, ascending=False)
    
    sns.barplot(x=feat_imp_df.importance.values, y = feat_imp_df.feature.values, orient='h')
    plt.tick_params(labelsize=30)
    plt.xlim(0,1.0)
    
    fig3 = plt.figure(figsize=(15,12)) 
    ax = sns.regplot(np.array(y_test), np.array(y_pred) , color='r', marker='o',\
                     scatter_kws={'alpha':0.3})
    plt.title("Random Forest", fontsize=40)
    plt.ylabel("Predicted energy production (kWh/yr)",fontsize=40)
    plt.xlabel("Real energy production (kWh/yr)",fontsize=40)
    plt.xlim(-500,1e5)
    plt.ylim(-500,1e5)
    plt.tick_params(labelsize=30)
    return
    

if __name__ == "__main__":
    data = read_data()
    data = encode_features(data)
    
    regressor, X_train, X_test, y_train, y_test = cross_validate(data)
    
    #Enable to do grid search of hyperparameters 
    #do_grid_search(regressor, X_train, y_train)
    
    y_pred, y_valid = make_predictions(regressor, X_train, X_test, y_train)
    dump_model(regressor)
    compute_error(y_test, y_pred)
    plot(regressor, y_train, y_valid, y_test, y_pred)
    
    
    
    
    
    
    
    
    
    
    