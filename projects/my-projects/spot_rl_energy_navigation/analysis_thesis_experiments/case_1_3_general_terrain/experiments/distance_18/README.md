# Selected eighteen-metre episodes

This is an outcome-selected illustrative collection, assembled manually from separate complete runs. It is not an unbiased random test sample and must not be used to estimate the policy's overall success rate. Complete source runs remain available separately.

`mission_list.json` is the authoritative registry with forty slots, numbered 1–40; `mission_list.csv` is its readable export. The current counts and confirmed/tentative split are exported automatically in `category_counts.csv` and `category_summary.json` after every addition. The intended category totals remain ten regular, ten ice, ten stairs and ten multiple-choice cases. No six-metre episode is included.

| Collection ID | Original episode | Original index within the 18 m group | Category |
| --- | --- | --- | --- |
| E1 | 64 | 24 | Stairs |
| E2 | 56 | 16 | Ice |
| E3 | 55 | 15 | Ice |

The source is `../distances_6m_18m_initial/results_exp/distance_18m_20261113_164333_391822`. All three recorded outcomes are goal completion. They are existing measurements, not new simulations.

Each `episodes/episode_XX/` directory contains the exact selected mission, original result row, provenance and episode-filtered CSV files for trajectory, power, diagnostics and result. The displayed episode field is renumbered; `source_episode` and `source_episode_18m` preserve the original IDs. Physical measurements, timestamps and coordinates are unchanged. Original endpoint IDs remain provenance identifiers, not collection numbering.

`sources/` keeps a copy of the complete source run's input snapshots, configuration, completion marker and summary. A source snapshot may include the original full training/evaluation coordinate panel; it does not add those other episodes to this collection. The aggregate `episodes.csv`, `trajectories.csv`, `power_samples.csv` and `fall_diagnostics.csv` contain only the selected episodes.

The checkpoint is pinned to the imported test7 `policy_best_stage_3.pt`, at 650,065 transitions. This checkpoint has the same weights as test7 `policy_best.pt`; it was trained up to curriculum stage 3 (6–12 m). Eighteen-metre evaluation extends beyond that training distance band. The model and evaluation map hashes are required to match when adding a source episode.

Use `add_episode.py --source-run PATH --source-episode ORIGINAL_ID` to add a manually chosen episode to the next free slot. Use `--slot N` to select a free collection slot explicitly. The helper refuses duplicate source episodes, occupied slots, incomplete runs or incompatible checkpoint/map hashes. Category overflow is refused by default; `--allow-category-overflow` permits retaining extra candidates when explicitly requested. Existing raw source files are read only.

`selected_missions.json` and the CSV exports are rebuilt from the registry after each addition. Run `plot_selected.py` with Isaac Sim's Python to regenerate `selected_routes.pdf/.png` and the per-episode PDF. E1, E2, etc. appear next to the start dots. These labels identify collection slots; labels in new test runs identify their own local episodes and are mapped explicitly when selected.

No thesis document, original result or success-rate analysis is modified here.

Use `--tentative` when adding an episode that is still under review. Such cases count as saved but are listed separately from confirmed selections. `review_status` is selection metadata, independent of the recorded physical outcome. Start labels use a larger bold sans-serif font in a white square box, keeping the episode color. The live plotting template follows this style for future attempts.

`review_episode.py N --remove --reason TEXT` removes a case from the active collection while retaining its files under `excluded/` and recording its selection history. The vacated slot becomes available; other IDs are not renumbered. `review_episode.py N --review confirmed` confirms a tentative selection. The category summaries are rebuilt after either action.

Original user notes are stored as `selection_note` in the registry and coordinate CSV. Notes "meh", "meh meh" and "mm" are tentative selection reviews; "fail" records the user note independently of the actual simulator outcome. The try2 Ep 54 failure is retained with its recorded `fallen` outcome. All eleven multiple-choice cases were explicitly retained for later selection; `category_counts.csv` reports the excess over the target separately.
