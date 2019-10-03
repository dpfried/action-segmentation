class Model(object):
    def fit(self, train_corpus):
        raise NotImplementedError()

    def predict(self, data):
        raise NotImplementedError()
