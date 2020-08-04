from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Model:
    def __init__(self, feature_matrix, labels, folds, cfg):

        self.X = feature_matrix
        self.encoder = LabelEncoder()
        self.y = self.encoder.fit_transform(labels)
        self.folds = folds
        self.cfg = cfg

        self.trained_models_ = []
        self.val_fold_scores_ = []

    def train_kfold(self):

        logo = LeaveOneGroupOut()
        for train_index, test_index in logo.split(self.X, self.y, self.folds):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            ss = StandardScaler(copy=True)
            X_train = ss.fit_transform(X_train)
            X_test = ss.transform(X_test)

            clf = self.cfg["model"]
            clf.fit(X_train, y_train)

            self.trained_models_.append(clf)
            y_pred = clf.predict(X_test)
            
            fold_acc = accuracy_score(y_test, y_pred)
            self.val_fold_scores_.append(fold_acc)
            print("logo fold complete")
            
            """
            pipe = Pipeline(
                [("scaler", StandardScaler(copy=True)), ("model", self.cfg["model"])]
            )

            kf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
            
            # TODO - This isn't valid. In this inner CV things things from same recording fold
            # could end up in train/test of the CV. What you could do is see if you could pass another
            # group-based splitter into GridSearchCV. It wouldn't be LOGO but rather GroupKFold
            # and you'd have to increase the inner CV splits. Alternatively, you'd simply not do a GridSearch.
            # That's maybe fine for the purposes here. Could still write a blog showing the pitfalls to avoid.
            # If you did try no GridSearch, the val set is really the mean score on the 10, and you're fitting to that,
            # if you choose between different model families, say. But then you'd want an external test set of random
            # example that you'd need to splice out. But maybe not even worth trying to max the RF. Could instead
            # spend the time seeing if Keras works on the non-GCV approach. I think I would just do a blog on what you have
            # here, and discuss the pitfalls, and then do another blog later trying to beat it. Compare it to the research paper.

            grid_search = GridSearchCV(
                estimator=pipe,
                param_grid=self.cfg["param_grid"],
                cv=kf,
                return_train_score=True,
                verbose=3,
                **self.cfg["grid_dict"]
            )

            grid_search.fit(X_train, y_train)
            self.trained_models_.append(grid_search.best_estimator_)

            X_test = grid_search.best_estimator_["scaler"].transform(X_test)
            y_pred = grid_search.best_estimator_["model"].predict(X_test)

            fold_acc = accuracy_score(y_test, y_pred)
            self.val_fold_scores_.append(fold_acc)
            """

        return self.val_fold_scores_
