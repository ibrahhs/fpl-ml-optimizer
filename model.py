from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

def train_predict(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(Xtr, ytr)
    preds = model.predict(X)
    return model, preds
