import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor
import pickle

def main():
    root_dir = './input/20240110_cubist_input_spi1'
    model_dir = './output/rfs'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    scores_list = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.data'):
                path = os.path.join(root, file)
                key = f"d{root.split('d')[-1]}"

                df = pd.read_csv(path, header=None)
                print(df.head())
                X = df.drop(columns=[6])
                y = df[6]
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=42)
                rf = RandomForestRegressor(random_state=42)
                rf.fit(X_train, y_train)

                y_train_pred = rf.predict(X_train)
                r2_train = r2_score(y_train, y_train_pred)

                y_test_pred = rf.predict(X_test)
                r2_test = r2_score(y_test, y_test_pred)


                with open(os.path.join(model_dir, f'rf_{key}.json'), 'wb') as f:
                    pickle.dump(rf, f)

                scores_list.append({'julian_date': root.split('d')[-1], 'training_data': round(r2_train, 2), 'test_data': round(r2_test, 2)})

    scores = pd.DataFrame(scores_list)
    scores.to_csv(os.path.join('./output', 'score_result.csv'), index=False)






if __name__ == '__main__':
    main()