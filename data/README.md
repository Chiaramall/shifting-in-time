# Dataset inputs

Supply prepared UTF-8 CSV files with a header and columns
`user_id,item_id,rating,timestamp` (comma-separated). Timestamps are Unix seconds;
IDs must be stable between training and evaluation. Rows are split chronologically
by the framework. Unknown test users/items are filtered by the dataset pipeline.

| Dataset | Expected path | Description |
| --- | --- | --- |
| LastFM-1K | `data/lastfm1k/user_events1k.csv` | Complete prepared event CSV; implicit rating 1.0 |
| MovieLens-10M | `data/movielens10m/movielens10m.csv` | Rating events with original timestamps |

Timestamps are read as 64-bit integers to preserve second-level chronological
ordering. The latest 10% of each history forms the test split; 10% of the
remaining history forms the validation split. Neither holdout is used for training.

The complete prepared LastFM CSV available in the source workspace contains
992 users, 960,402 distinct tracks, and 16,936,136 interactions. Both LastFM
configurations read this full file; there is no 30,000-track cap. These counts
refer to the prepared CSV, before temporal splitting and unknown-item filtering,
not necessarily to every raw event in the original TSV.

MovieLens `ratings.dat` can be converted to the required schema by reading its
`user::movie::rating::timestamp` fields and writing the four CSV columns above.
No dataset is bundled or downloaded automatically.
