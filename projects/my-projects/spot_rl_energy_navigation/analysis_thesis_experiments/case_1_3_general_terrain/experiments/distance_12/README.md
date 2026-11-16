# Selected twelve-metre episodes

This collection contains the requested illustrative cases. `mission_list.json` grows to accommodate extra candidates; `mission_list.csv` is the readable export. The initial target is ten selected cases per category: regular, ice, stairs and surface_choice. `category_counts.csv` and `category_summary.json` are rebuilt after every selection/review.

The complete trials remain separate in sibling folders such as `../distance_12m_try1/`. Selecting episodes after seeing outcomes produces an illustrative collection, not an unbiased random test sample or a basis for overall success-rate estimates. No trial outcome is removed from its complete source run.

Use `add_episode.py --source-run PATH --source-episode ID` to collect an explicitly selected completed twelve-metre episode. `--tentative` marks a case still under review; `--note TEXT` retains the original selection note. Add `--allow-category-overflow` only when extra candidates have explicitly been requested. Raw source measurements remain unchanged; the source model and map hashes must match the collection.

`episodes/episode_XX/` will retain the mission, result, filtered trajectory/power/diagnostic logs and provenance. `sources/` will preserve source input snapshots, configuration, summary and completion marker. Renumbered collection IDs always retain their source run and source episode identifiers. No six- or eighteen-metre case can be added.

Use `review_episode.py N --review confirmed` to confirm a tentative case. `review_episode.py N --remove --reason TEXT` archives an excluded selection without deleting its measurements. Other collection IDs are not renumbered. `plot_selected.py` regenerates selected route figures once cases exist; labels use colored sans-serif E boxes with the source try1 episode number. Legend parentheses identify the collection number.

The policy is test7 `policy_best_stage_3.pt`; the map is shared in `../../maps/`. No episode is selected automatically, and no thesis document or aggregate performance analysis is changed by these tools.

## Try1 selection — 2026-11-16

All 92 requested episodes are archived (regular 28, ice 24, stairs 22, surface_choice 18), including all five requested falls. The final table target remains ten per category; extra candidates are kept. The fourth category is named surface_choice in the source twelve-metre experiment, corresponding to the user list labeled multiple_choice.

`selection_request_try1.json` preserves the exact requested episode list, notes and bold highlights. The 19 bold-highlighted cases are marked `preferred_for_plot` in the registry, CSV and per-episode metadata, and listed separately in `preferred/mission_list.json` and `.csv`. These records point to the main archived episode data; raw measurements are not duplicated. The final choice of two displayed routes per category is still open.

Highlighted try1 IDs:

- Regular: 3, 59, 83.
- Ice: 204, 246, 259, 273, 280, 317, 333.
- Stairs: 411, 453, 479, 567, 597.
- Surface choice: 624, 633, 639, 707.

`storto` and `stortino` denote the user's observation of a curved path despite reaching the goal. A straight path might consume less energy, but the comparison has not been measured. Notes `meh`, `rampina`, `rampa`, `ostacoli` and `ondulato` are retained. Recorded outcomes are preserved separately from these comments.

`selected_routes.pdf` and `.png` show all selected cases in four category panels. `preferred_routes.pdf` and `.png` show only the 19 highlighted candidates in the same layout. `selected_routes` labels show collection IDs E1–E92, with original try1 IDs in legend parentheses. `preferred_routes` retains the original try1 labels and collection IDs in legend parentheses. `selected_routes_per_episode.pdf` provides one page per archived case; `preferred/routes_per_episode.pdf` provides one page per highlighted candidate. Run `plot_selected.py` to regenerate these plots. All panels include falls if selected; no training or original result file was changed.
