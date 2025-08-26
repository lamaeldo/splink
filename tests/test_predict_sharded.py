import pandas as pd
from pandas.testing import assert_frame_equal

from splink.internals.duckdb.database_api import DuckDBAPI
from splink.internals.linker import Linker
from tests.basic_settings import get_settings_dict

from .decorator import mark_with_dialects_including


@mark_with_dialects_including("duckdb")
def test_predict_sharded_matches_predict():
    df = pd.read_csv("./tests/datasets/fake_1000_from_splink_demos.csv")
    df1 = df.iloc[:50].copy()
    df2 = df.iloc[50:100].copy()

    settings = get_settings_dict()
    settings["link_type"] = "link_only"
    settings["source_dataset_column_name"] = "source_dataset"

    linker = Linker([df1, df2], settings, DuckDBAPI())
    expected = linker.inference.predict().as_pandas_dataframe()

    linker_sharded = Linker([df1, df2], settings, DuckDBAPI())
    result = (
        linker_sharded.inference.predict_sharded(num_shards=3).as_pandas_dataframe()
    )

    expected = expected.sort_values(["unique_id_l", "unique_id_r"]).reset_index(
        drop=True
    )
    result = result.sort_values(["unique_id_l", "unique_id_r"]).reset_index(
        drop=True
    )

    assert_frame_equal(expected, result)
