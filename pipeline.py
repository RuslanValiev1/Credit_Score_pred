
import dill
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, make_scorer, confusion_matrix
from sklearn.pipeline import Pipeline


def main():
    data = pd.read_pickle('train_data/train_data.pkl')
    model = lgb.LGBMClassifier(class_weight='balanced')
    x = data.drop(columns=['flag']).astype('int8')
    y = data['flag']

    categorical_features = make_column_selector()
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, categorical_features)
    ])

    preprocessor = Pipeline(steps=[
        ('column_transformer', column_transformer)
    ])

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipe.fit(x, y)

    with open('def_predict_model.pkl', 'wb') as file:
        dill.dump({
            'model': pipe,
            'metadata': {
                'name': 'default prediction model',
                'author': 'VALIEV Ruslan',
                'version': 1,
                }
        }, file)

if __name__ == '__main__':
    main()






















# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()