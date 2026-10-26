# Tre mappe aggiuntive per la policy test 13

Legenda: A = asfalto, R = rocce originali del test 13, I = ghiaccio.
Celle 2×2 m, centri x/y a 0, 2, 4 m. Le righe seguono la convenzione
del generatore originale: la prima è y=0, la seconda y=2, la terza y=4;
le colonne crescono lungo x. Quindi, viste dall'alto con +y verso l'alto,
la prima riga riportata qui si trova in basso.
Partenza (0,0), goal (4,4), come nel test 13.

`new_map_1.usd`:

```text
A R R
A I A
A R A
```

`new_map_2.usd`:

```text
A R A
A I A
R R A
```

`new_map_3.usd`:

```text
A R A
R I R
A A A
```

Le celle sono copie traslate di quelle presenti in `rocks_right.usd`:
stesse mesh, altezze, collisioni, attriti e materiali. Le celle di uno stesso
tipo possono riutilizzare lo stesso donatore. Nessuna generazione casuale
o modifica dei terreni. Donatori e hash sono in `new_maps_manifest.json`.
Le due mappe di training e l'identità richiesta dai checkpoint sono conservate;
queste tre mappe sono registrate solo per la valutazione.

Per valutarle tutte e tre, una alla volta e in sequenza automatica, dalla
radice del progetto:

```bash
bash spot_rl_energy_navigation/multi_terrain_test13/play_new_maps.sh \
  --checkpoint spot_rl_energy_navigation/multi_terrain_test13/runs/test13_ppo_20261023_122929_726370/policy_final.pt \
  --episodes 5
```

Per una sola mappa:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p \
  spot_rl_energy_navigation/multi_terrain_test13/play.py \
  --checkpoint spot_rl_energy_navigation/multi_terrain_test13/runs/test13_ppo_20261023_122929_726370/policy_final.pt \
  --layout new_map_1 --episodes 5 --device cuda:0 --agent-device cpu
```

Risultati in sottocartelle distinte di `../results/`: successo, energia,
percorso e traiettorie. In tutte e tre queste mappe entrambi i rami laterali
contengono rocce: `asphalt_route` è quindi zero e non indica quale percorso
sia migliore. Per confrontare i percorsi usare le traiettorie e i joule degli
episodi riusciti; non inferire il costo energetico dal solo lato scelto.

Per ricreare gli USD usando gli stessi donatori (senza avviare Isaac Sim):

```bash
/home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh \
  spot_rl_energy_navigation/multi_terrain_test13/create_test_maps.py --overwrite
```
