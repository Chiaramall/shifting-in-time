# Dataset inputs

Supply prepared UTF-8 CSV files with a header and columns
`user_id,item_id,rating,timestamp` (comma-separated). Timestamps are Unix seconds;
IDs must be stable between training and evaluation. Rows are split chronologically
by the framework. Unknown test users/items are filtered by the dataset pipeline.

| Dataset | Expected path | Description |
| --- | --- | --- |
| LastFM-1K | `data/lastfm1k/user_events_30000.csv` | Original experiment's 30,000-track subset; implicit rating 1.0 |
| MovieLens-10M | `data/movielens10m/movielens10m.csv` | Rating events with original timestamps |

The LastFM YAML preserves the original float32 timestamp conversion. Changing it
can alter ordering of near-simultaneous events and the resulting split.

The original LastFM preprocessing scripts referenced intermediate CSV files and
incompatible ID mappings. They have not been presented as an exact raw-to-paper
recipe. Obtain the prepared subset from the experiment maintainers to reproduce
its results; otherwise report the new preprocessing and subset explicitly.

MovieLens `ratings.dat` can be converted to the required schema by reading its
`user::movie::rating::timestamp` fields and writing the four CSV columns above.
No dataset is bundled or downloaded automatically.
