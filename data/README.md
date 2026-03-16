# Data instructions

Download a MovieLens dataset and extract it under this directory.

Recommended for fast iteration:
- `ml-latest-small`

Expected structure:

```text
data/ml-latest-small/
├── ratings.csv
└── movies.csv
```

Minimal required files:
- `ratings.csv`
- `movies.csv`

The project assumes `ratings.csv` contains at least:
- `userId`
- `movieId`
- `rating`
- `timestamp`

and `movies.csv` contains:
- `movieId`
- `title`
- `genres`
