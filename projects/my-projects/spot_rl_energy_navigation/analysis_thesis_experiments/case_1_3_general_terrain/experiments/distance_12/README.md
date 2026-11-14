# Selected twelve-metre episodes

This collection starts empty. `mission_list.json` contains forty slots, numbered E1–E40, and `mission_list.csv` is the readable export. The initial target is ten selected cases per category: regular, ice, stairs and surface_choice. `category_counts.csv` and `category_summary.json` are rebuilt after every selection/review.

The complete trials remain separate in sibling folders such as `../distance_12m_try1/`. Selecting episodes after seeing outcomes produces an illustrative collection, not an unbiased random test sample or a basis for overall success-rate estimates. No trial outcome is removed from its complete source run.

Use `add_episode.py --source-run PATH --source-episode ID` to collect an explicitly selected completed twelve-metre episode. `--tentative` marks a case still under review; `--note TEXT` retains the original selection note. Add `--allow-category-overflow` only when extra candidates have explicitly been requested. Raw source measurements remain unchanged; the source model and map hashes must match the collection.

`episodes/episode_XX/` will retain the mission, result, filtered trajectory/power/diagnostic logs and provenance. `sources/` will preserve source input snapshots, configuration, summary and completion marker. Renumbered collection IDs always retain their source run and source episode identifiers. No six- or eighteen-metre case can be added.

Use `review_episode.py N --review confirmed` to confirm a tentative case. `review_episode.py N --remove --reason TEXT` archives an excluded selection without deleting its measurements. Other collection IDs are not renumbered. `plot_selected.py` regenerates selected route figures once cases exist; labels use the requested larger colored sans-serif E boxes.

The policy is test7 `policy_best_stage_3.pt`; the map is shared in `../../maps/`. No episode is selected automatically, and no thesis document or aggregate performance analysis is changed by these tools.
