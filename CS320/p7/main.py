# project: p7
# submitter: hseverson4
# partner: none
# hours: 8

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression


class UserPredictor():
    def __init__(self):
        self.pipe = Pipeline([
    ("poly", PolynomialFeatures(2)), 
    ("std", StandardScaler()),
    ("lr", LogisticRegression()),
])
    
    def fit(self, train_users, train_logs, train_y):
        y_plus_users = train_users.join(train_y, on="user_id",rsuffix="_other")
        y_plus_users = y_plus_users.drop(columns="user_id_other")
        
        laptop = train_logs[train_logs['url']=='/laptop.html']
        laptop_secs = laptop[['user_id', 'seconds']].groupby(['user_id']).sum()
        
        final = y_plus_users.join(laptop_secs, on="user_id", rsuffix="other").fillna(0)
        final["laptop"] = final["seconds"] > 0
        
        final["gold"] = final["badge"] == "gold"

        self.pipe.fit(final[["past_purchase_amt", "age", "seconds", "laptop", "gold"]], final["y"])
        

    def predict(self, test_users, test_logs):
        test_laptop = test_logs[test_logs['url']=='/laptop.html']
        test_lap_secs = test_laptop[['user_id', 'seconds']].groupby(['user_id']).sum()
        
        test_final = test_users.join(test_lap_secs, on="user_id", rsuffix="other").fillna(0)
        test_final["laptop"] = test_final["seconds"]>0
        
        test_final["gold"] = test_final["badge"] == "gold"
        
        return self.pipe.predict(test_final[["past_purchase_amt", "age", "seconds", "laptop", "gold"]])
    
    
    
    
    
    
    
    