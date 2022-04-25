from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

class RegALL():

    def Logreg(self, X_train, X_test, y_train):
        self.logr = LogisticRegression()
        self.logr.fit(X_train, y_train.astype(int))
        self.pred = self.logr.predict(X_test)
        return self.pred

    def Linreg(self, X_train, X_test, y_train):
        self.linr = LinearRegression()
        self.linr.fit(X_train, y_train)
        self.pred = self.linr.predict(X_test)
        return self.pred
    
    def Rid(self, X_train, X_test, y_train, alpha=10):
        self.rid = Ridge(alpha=alpha)
        self.rid.fit(X_train, y_train)
        self.pred = self.rid.predict(X_test)
        return self.pred
    
    def Las(self, X_train, X_test, y_train, alpha=0.01):
        self.las = Lasso(alpha=alpha)
        self.las.fit(X_train, y_train)
        self.pred = self.las.predict(X_test)
        return self.pred

    def Elastic(self, X_train, X_test, y_train, alpha=1):
        self.elar = ElasticNet(alpha=alpha)
        self.elar.fit(X_train, y_train)
        self.pred = self.elar.predict(X_test)
        return self.pred