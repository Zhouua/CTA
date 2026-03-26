from os import path

import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(
            file_path,
            dtype={"CONTRACT": str, "CONTRACTID": str},
            index_col=0,
        )
        self.data["TDATE"] = pd.to_datetime(self.data["TDATE"])
        self.data = self.data.sort_values("TDATE").reset_index(drop=True)

        self.train = None
        self.val = None
        self.test = None

        self.cal_return()

    def cal_return(self, horizon=5):
        self.data["5min_return"] = np.log(
            self.data["CLOSE"].shift(-horizon) / self.data["CLOSE"]
        )
        return self.data

    def split_data(self, train_ratio, val_ratio, test_ratio):
        print("Splitting data...")
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

        train_size = int(train_ratio * len(self.data))
        val_size = int(val_ratio * len(self.data))

        self.train = self.data.iloc[:train_size].copy()
        self.val = self.data.iloc[train_size : train_size + val_size].copy()
        self.test = self.data.iloc[train_size + val_size :].copy()

        print(f"Train Date: {self.train['TDATE'].iloc[0]} to {self.train['TDATE'].iloc[-1]}")
        print(f"Validation Date: {self.val['TDATE'].iloc[0]} to {self.val['TDATE'].iloc[-1]}")
        print(f"Test Date: {self.test['TDATE'].iloc[0]} to {self.test['TDATE'].iloc[-1]}")
        return self.train, self.val, self.test


if __name__ == "__main__":
    file_path = path.join("data", "RBZL.SHF.csv")
    data_loader = DataLoader(file_path)
    print(data_loader.data.head(20))
    train, val, test = data_loader.split_data(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
