# Eighteen-metre thesis figures

Forty manually selected illustrative cases, ten per category, from the archived `../distance_18/` collection. The source collection, previous figures and raw measurements remain unchanged. This subset is not an unbiased random evaluation sample.

## Figures

All figures have PDF and PNG versions:

- `exp_all_together`: forty cases in four category panels.
- `exp_18m_cat1_regular`: E1–E10.
- `exp_18m_cat2_ice`: E11–E20.
- `exp_18m_cat3_stairs`: E21–E30.
- `exp_18m_cat4_multiple_choices`: E31–E40.
- `show_routes`: eight user-selected display examples, two per category.

Numbering follows category order, then ascending old collection ID within each category. Plot labels use the new thesis IDs. Legends temporarily retain the old collection IDs (the numbers used in the user's selection). `episode_numbering.csv` also records source run, original source episode, original category and outcome.

The display palette matches the twelve-metre figures: Regular green, Ice blue, Stairs orange, Multiple choices fuchsia. Category panels use the same ten episode colors as the twelve-metre panels. Every requested failure keeps its recorded outcome and terminal marker.

## Selection and presentation categories

The user-requested transfers are applied only to `thesis_category` in this folder. Original mission/category evidence, recorded results and source provenance remain intact under `case`, `result` and `original_category` in `selected_missions.json`.

- Old E42: Multiple choices to Ice.
- Old E10, E24, E28 and E32: Multiple choices to Stairs.
- Discarded from this subset: old E11, E16, E17, E25, E27, E30, E41 and E46. They remain archived in the original collection.

- regular: old E4, old E6, old E7, old E12, old E13, old E14, old E15, old E33, old E34, old E35.
- ice: old E2, old E3, old E5, old E18, old E19, old E20, old E21, old E22, old E36, old E42.
- stairs: old E1, old E8, old E10, old E23, old E24, old E28, old E32, old E37, old E38, old E39.
- multiple_choices: old E9, old E26, old E29, old E31, old E40, old E43, old E44, old E45, old E47, old E48.

## Display examples

`show_routes.json` lists the eight selected display cases and their old, thesis and source IDs. Old E6 and E13 are shown for Regular. Optional old E34 (out_of_bounds) remains in the forty-case Regular panel, but is not added to the eight-case display.

- regular: thesis E2, old E6.
- regular: thesis E5, old E13.
- ice: thesis E12, old E3.
- ice: thesis E19, old E36.
- stairs: thesis E28, old E37.
- stairs: thesis E23, old E10.
- multiple_choices: thesis E32, old E26.
- multiple_choices: thesis E40, old E48.

## Pending cleanup

Remove old episode IDs from legend parentheses for the final thesis figures when requested. Keep provenance mappings. The same reminder is recorded for the twelve-metre figures.

`plot_routes.py` regenerates all six figures using original archived trajectories. It validates map identity, category membership in the selected grouping, terminal samples and energy consistency.
