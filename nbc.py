import numpy as np

# 定义多项式朴素贝叶斯


class MultinomialNB(object):

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.classes = None
        self.conditional_prob = None

    # 计算
    def _calculate_feature_prob(self, feature):
        values = np.unique(feature)
        total_num = float(len(feature))
        value_prob = {}
        for v in values:
            value_prob[v] = ((np.sum(np.equal(feature, v)) + self.alpha) / (total_num + len(values) * self.alpha))
        return value_prob

    # 模型训练
    def fit(self, X, y):
        # TODO: check X,y

        self.classes = np.unique(y)
        # 计算每一类的先验概率 P(y=ck)
        if self.class_prior == None:
            class_num = len(self.classes)
            if not self.fit_prior:
                self.class_prior = [1.0 / class_num for _ in range(class_num)]  # uniform prior
            else:
                self.class_prior = []
                sample_num = float(len(y))
                for c in self.classes:
                    c_num = np.sum(np.equal(y, c))
                    self.class_prior.append((c_num + self.alpha) / (sample_num + class_num * self.alpha))

        # 计算条件概率 P( xj | y=ck )
        self.conditional_prob = {}  # like { c0:{ x0:{ value0:0.2, value1:0.8 }, x1:{} }, c1:{...} }
        for c in self.classes:
            self.conditional_prob[c] = {}
            for i in range(len(X[0])):  # for each feature
                feature = X[np.equal(y, c)][:, i]
                self.conditional_prob[c][i] = self._calculate_feature_prob(feature)
        return self

    # given values_prob {value0:0.2,value1:0.1,value3:0.3,.. } and target_value
    # return the probability of target_value
    def _get_xj_prob(self, values_prob, target_value):
        if target_value not in values_prob:
            return 0.01
        return values_prob[target_value]

    # 基于先验概率和条件概率预测样本
    def _predict_single_sample(self, x):
        label = -1
        max_posterior_prob = 0

        # 对于每一类，计算后验概率：先验概率*条件概率
        for c_index in range(len(self.classes)):
            current_class_prior = self.class_prior[c_index]
            current_conditional_prob = 1.0
            feature_prob = self.conditional_prob[self.classes[c_index]]
            j = 0
            for feature_i in feature_prob.keys():
                current_conditional_prob *= self._get_xj_prob(feature_prob[feature_i], x[j])
                j += 1

            # 比较后验概率 更新最大后验概率和标签
            if current_class_prior * current_conditional_prob > max_posterior_prob:
                max_posterior_prob = current_class_prior * current_conditional_prob
                label = self.classes[c_index]
        return label

    # 预测样本集合的标签
    def predict(self, X):
        # TODO1:check and raise NoFitError
        # ToDO2:check X
        if X.ndim == 1:
            return self._predict_single_sample(X)
        else:
            # 对每一个样本的类别进行预测
            labels = []
            for i in range(X.shape[0]):
                label = self._predict_single_sample(X[i])
                labels.append(label)
            return labels