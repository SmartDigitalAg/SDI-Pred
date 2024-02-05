import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor
import pickle

def main():
    model_dir = './output/rfs'
    inputs_path = './input/output_건천리_추가.xlsx'
    input = pd.read_excel(inputs_path)
    input = input[input['FID'] == 10]

    print(input.columns)
    all_model = [os.path.join(model_dir, path) for path in os.listdir(model_dir) if path.endswith('.json')]



    result = pd.DataFrame()
    for idx, week in enumerate(input['Week'].unique()):
        model_path = all_model[idx]
        with open(model_path, 'rb') as f:
            rf = pickle.load(f)

        each_week = input[input['Week'] == week]
        X = each_week[['DEM', 'LC', 'AWC', 'ECO', 'SSG', 'SPI1']]
        each_week["SDI"] = rf.predict(X)

        result = pd.concat([result, each_week])

    result = result.sort_values(by=['Year', 'Week'])
    result.to_csv('./output/pred_result.csv', index=False)



    #
    #
    # for root, dirs, files in os.walk(model_dir):
    #     for file in files:
    #         if file.endswith('.json'):
    #             path = os.path.join(root, file)
    #             # with open(path, 'rb') as f:
    #             #     rf = pickle.load(f)
    #
    #
    #             # preds = rf.predict(new_X)


if __name__ == '__main__':
    main()