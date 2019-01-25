import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re


# 디렉토리 안에 모든 파일들을 DataFrame 형태로 읽어옵니다.
# 구체적으로 문장(sentence)과 문장의 감정상태의 확신정도(sentiment=1~10)를 읽어옵니다.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


# 긍정(postive) 예제와 부정(negative) 예제를 하나의 dataframe으로 합치고
# 긍정 혹은 부정을 나타내는 polarity 컬럼을 추가하고 데이터를 랜덤하게 섞습니다.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# IMDB 영화 리뷰 데이터셋을 다운받고 전처리를 진행합니다.
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True)

    train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                         "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                        "aclImdb", "test"))

    return train_df, test_df


# 로깅 레벨을 ERROR로 설정합니다.
tf.logging.set_verbosity(tf.logging.ERROR)

train_df, test_df = download_and_load_datasets()
train_df.head()

# 전체 트레이닝 데이터로 트레이닝 셋을 구성합니다.
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], num_epochs=None, shuffle=True)

# 전체 트레이닝 데이터셋에 대해 에측을 진행합니다.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], shuffle=False)
# 테스트 데이터셋에 대해 예측을 진행합니다.
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, test_df["polarity"], shuffle=False)

# nnlm-en-dim128 모듈을 이용해서 Pre-Trained 모델을 불러와서 임베딩을 수행합니다.
embedded_text_feature_column = hub.text_embedding_column(
    key="sentence",
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

# Text Classification을 수행하기 위한 DNNClassifier Estimator를 정의합니다.
estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

# 128,000개의 트레이닝 데이터를 이용해서 1000번 학습을 진행합니다.
estimator.train(input_fn=train_input_fn, steps=1000);

# 트레이닝 데이터와 테스트 데이터에 대해서 모델의 정확도를 출력합니다.
train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Test set accuracy: {accuracy}".format(**test_eval_result))