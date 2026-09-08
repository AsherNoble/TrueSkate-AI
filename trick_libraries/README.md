# Retained gesture recipes

These recipes are useful outputs of retired CMA-ES optimisation. BC's
`data.gesture_sampling.load_recipe_vectors` reads them for recipe sampling;
`scripts/inspect/execute_trick.py` replays them. They retain their existing
filenames and schema. No live optimiser is required to consume them.

`GESTURES.md` defines normalised coordinates, curved waypoints, duration,
easing, delays and the optional spin hold. Values are gesture parameters,
not model weights or probabilities. Median and best variants reflect mining
choices, not guarantees that every replay lands the named trick.

The original libraries, curricula, mining tools and tracked source logs are
recoverable at `archive/research-pre-cleanup` (ARCH-002). `scripts/data/build_trick_library.py`
and `mine_all_tricks.py` remain available for inspecting/mining historical logs.
Individual JSON provenance fields are retained as supplied. Where an exact
generating SHA/configuration is absent, provenance is incomplete; do not invent it
or treat the archive snapshot SHA as the generating version.
