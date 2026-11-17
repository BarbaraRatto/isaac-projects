# Confirmed straight-line results

The current main-document comparisons use the remote results confirmed by the user on 17 November 2026. Their supplied source tables are `straight_line_command_mean_12m.tex` (former document tables 26–29) and `straight_line_command_mean_18m.tex` (former tables 30–33).

`big_map_12m_rettilineo_confermato.json` and `big_map_18m_rettilineo_confermato.json` freeze the exact supplied table rows, source hashes, paired policy records and provenance. The full-precision remote results are not available locally. Baseline statistics therefore use displayed path, energy and time values; policy statistics use archived full-precision measurements validated against the existing policy tables. Failed episodes are excluded from energy, distance and time averages. Relative energy and path comparisons use only missions successful with both methods.

`genera_tabelle_rettilineo.py` now reads these confirmed snapshots; it does not use the earlier local baseline logs. The two source-table generators restore the confirmed remote tables whenever a snapshot exists, to avoid replacing them with earlier local measurements. Original local runs remain archived separately.

Ten displayed mean speeds were corrected with user approval on 17 November 2026: three at 12 m and seven at 18 m. Corrected speeds are displayed path divided by displayed total time, rounded to two decimals. Snapshots preserve the original source tables, previous values and changed rows in original_source_table_content and speed_corrections. Both main-text and end-of-document tables use the corrected values. Energy, path, time and outcome comparisons are unchanged.

The old source tables remain at the end of the document for reference. Main-text tables retain their original labels and numbering.

On 17 November 2026 the user supplied four updated 12 m category-1 straight-line energies: E2 2432.5 J, E3 2595.5 J, E9 2095.3 J and E10 2393.4 J. They are recorded in energy_revisions in the confirmed snapshot; both table copies and energy comparisons use these updated values.

The subsequent user-provided 12 m category-3 energies are E21 2398.4 J and E28 2687.0 J. The snapshot records both previous values; result tables, comparison statistics and text use these updates.

Latest user-confirmed category-1 straight-line energies: E1 2534.2 J; E2 2465.5 J; E4 2498.9 J; E5 2474.8 J; E6 2631.3 J; E7 2664.7 J; E9 2186.3 J; E10 2431.4 J. Previous values are retained in energy_revisions.

Latest category-4 update: E32 2274.3 J; E33 2586.9 J; E35 2491.2 J; E36 2557.8 J; E37 2479.0 J; E38 2585.1 J. E31 was listed without a new value and retains 2593.1 J pending user input.

Latest 18 m category-1 straight-line update: E1 3791.1 J; E4 3783.6 J; E5 3754.7 J; E6 4031.5 J; E7 3708.4 J; E9 3578.0 J. E3 and E8 remain unchanged. All revisions are stored in the 18 m confirmed snapshot.

Updated 18 m category-3 straight-line energies: E23 3651.8 J and E24 3974.6 J. Previous values remain in the snapshot energy_revisions.

Updated 18 m category-4 straight-line energies: E33 3819.5 J and E36 3841.4 J. Previous values remain in the snapshot energy_revisions.
