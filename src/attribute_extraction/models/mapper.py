import json

import pandas as pd


class Mapper:

    UNK = "[UNK]"

    def __init__(self, attribute_code_col: str, attribute_value_col: str, mappings: dict = None):

        self.attribute_code_col = attribute_code_col
        self.attribute_value_col = attribute_value_col
        self.mappings = mappings

    def fit(self, data):
        self.mappings = {}

        attributes_to_values = (
            data.groupby(by=self.attribute_code_col)[self.attribute_value_col].agg(list).to_dict()
        )

        for attribute_code, values in attributes_to_values.items():
            sorted_values = sorted(set([x for x in values if x is not None]))

            self.mappings[attribute_code] = {
                "id2label": {
                    0: self.UNK,
                    **{i + 1: label for i, label in enumerate(sorted_values)},
                },
                "label2id": {
                    self.UNK: 0,
                    **{label: i + 1 for i, label in enumerate(sorted_values)},
                },
            }

    def map_dataframe(self, data: pd.DataFrame, mapped_col_name: str):

        data[mapped_col_name] = data.apply(
            lambda x: self.mappings[x[self.attribute_code_col]]["label2id"].get(
                x[self.attribute_value_col], 0
            ),
            axis=1,
        )

        return data

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, dict_mapper: dict):
        return cls(**dict_mapper)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, sort_keys=True, indent=4, ensure_ascii=False)

    @classmethod
    def load(cls, path):

        with open(path, "r") as f:
            dict_mapper = json.load(f)
            for attribute_code, mappings in dict_mapper["mappings"].items():
                mappings["id2label"] = {int(k): v for k, v in mappings["id2label"].items()}

            return cls.from_dict(dict_mapper)


if __name__ == "__main__":

    df = pd.DataFrame(
        {
            "attribute_code": ["001", "002", "002", "003", "001"],
            "lov_code": ["a", "x", "y", "q", "b"],
        }
    )

    mapper = Mapper(attribute_code_col="attribute_code", attribute_value_col="lov_code")

    mapper.fit(df)

    print(mapper.mappings)
