from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import joblib,os


### fetching data

categories = [
        'rec.motorcycles',
        'rec.sport.baseball',
        'sci.electronics',
        'sci.space',
        'soc.religion.christian',
]

data_train = fetch_20newsgroups(
    subset="train",
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes"),
)

data_test = fetch_20newsgroups(
    subset="test",
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes"),
)

print(f"\nLoading 20 newsgroups dataset for {len(data_train.target_names)} categories: {data_train.target_names}")

print(f'\nTrain Records: {len(data_train.data)}')
print(f'Test Records: {len(data_test.data)}')


### creating sklearn pipeline
pipeline = Pipeline(
    [
        ("vect", TfidfVectorizer(stop_words='english', max_features=1000)),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)

print(f'\nTraining model')
pipeline.fit(data_train.data, data_train.target)

print(f'\nEvaluating model')
train_predictions = pipeline.predict(data_train.data)
test_predictions = pipeline.predict(data_test.data)
print(f"Train F1 score : {round(f1_score(data_train.target, train_predictions, average='micro'),2)}")
print(f"Test F1 score : {round(f1_score(data_test.target, test_predictions, average='micro'),2)}")

PATH = 'models/model.joblib'
print(f'\nSaving Model to location : {PATH}\n')

if not os.path.exists(PATH.split('/')[0]):
    os.makedirs(PATH.split('/')[0])
joblib.dump(pipeline, PATH)
