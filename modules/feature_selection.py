from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def select_features(x, y):
    # Feature extraction
    model = LogisticRegression()
    rfe = RFE(model, 30)
    fit = rfe.fit(x, y)
    print("Num Features: %s" % fit.n_features_)
    print("Selected Features: %s" % fit.support_)
    print("Feature Ranking: %s" % fit.ranking_)
