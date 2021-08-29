import pandas as pd


def read_keel_as_pandas(keel_path):
    with open(keel_path) as f:
        content = f.read().splitlines()

    attributes = [
        l.split(" ")[1] for l in content if l.lower().strip().startswith("@attribute")
    ]

    data = [l for l in content if not l.strip().startswith("@")]

    dict_format_rows = [dict(zip(attributes, row.split(","))) for row in data]

    return pd.DataFrame(dict_format_rows)
