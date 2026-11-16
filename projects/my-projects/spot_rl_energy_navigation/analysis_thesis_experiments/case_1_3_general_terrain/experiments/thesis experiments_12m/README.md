# Twelve-metre thesis figures

This folder contains forty manually selected illustrative episodes, ten per category. The original `../distance_12/` collection, raw measurements and figures remain unchanged. This subset is used for illustration, not as an unbiased random evaluation sample.

## Numbering

Thesis IDs follow the chosen order in `selected_missions_v2.json`:

- Regular: E1–E10.
- Ice: E11–E20.
- Stairs: E21–E30.
- Surface choice: E31–E40.

All six figures use these IDs, including `show_routes`. Legends also give original try1 IDs. `episode_numbering.csv` maps thesis, collection and original IDs. Archived measurements keep their original collection IDs.

Collection Regular E27 appeared in both discard and keep lists. The explicit keep list was followed; this case is thesis E9.

## Figures

- `exp_all_together.pdf` and `.png`: all forty episodes in four category panels, replacing the former `selected_routes_v2` filename.
- `exp_12m_cat1_regular.pdf` and `.png`: E1–E10.
- `exp_12m_cat2_ice.pdf` and `.png`: E11–E20.
- `exp_12m_cat3_stairs.pdf` and `.png`: E21–E30.
- `exp_12m_cat4_surface_choice.pdf` and `.png`: E31–E40.
- `show_routes.pdf` and `.png`: the same ten display examples as before, relabeled consistently. Regular is green, Ice blue, Stairs orange and Surface choice fuchsia.

`show_routes.json` preserves thesis, collection and try1 IDs. `show_routes_provenance.json` records original-copy history and current output hashes. `selected_missions_v2.json` freezes all forty missions and results with the source registry hash.

`plot_routes.py` regenerates all six figures, reading archived trajectories. It verifies category membership, terminal states and energy consistency. Starts use boxed thesis IDs, stars mark assigned goals and crosses mark failed terminal positions. Failed cases retain their recorded outcomes in legends.

## Pending figure cleanup

Remove old episode IDs in legend parentheses for the final thesis figures, when requested. Keep the mapping and archived provenance.
