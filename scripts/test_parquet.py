import polars as pl
from pathlib import Path

CURRENT: Path = Path('.')
RESULTS: Path = CURRENT / 'results'
data_path = RESULTS / 'network_performance_old.parquet'

data = pl.read_parquet(data_path)

print(data)
