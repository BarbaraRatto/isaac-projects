# Scelta energetica: rocce → rocce, rampa oppure asfalto

Questo esperimento è separato da `ice_choice`. Il USD originale non viene modificato.
`build_scene.py` compone `ramp_energy_choice.usda` usando le celle originali
`Cell_1_3_t5_large_rocks`, `Cell_1_4_t3_ramp`, `Cell_1_5_t5_large_rocks` e
una deviazione continua di asfalto nella riga 0, colonne 3–5. Nel USD
originale le celle 0,3 e 0,5 erano rispettivamente ghiaia e scale: la variante
le sostituisce con copie della cella asfaltata 0,4, mantenendo texture,
collisioni e attriti dell'asfalto originale.

- Partenza: `(19,5, 3)` nelle rocce, a metà fra il centro della cella e il bordo verso la rampa; orientamento `+X`.
- Goal: `(28,5, 3)` nelle rocce, a metà fra il bordo vicino alla rampa e il centro della cella; raggio di arrivo `0,55 m`.
- Via diretta: rampa 7° al centro `(24, 3)`.
- Deviazione: corridoio asfaltato `-1,5 <= Y <= 1,5`, a destra di Spot.
- 32 copie indipendenti della scena e 32 camere ZED X RGB-D 640×360; un
  DINOv2-small congelato condiviso, elaborato in microbatch di 4 immagini.
- Le immagini vengono elaborate live. `../runs/ice_choice_atlas.npz` fornisce
  **solo** la base PCA per comprimere le feature DINO, non un costo o una
  classe di terreno e non le feature memorizzate delle celle viste dalla policy.
- La policy sceglie `vx`, `vy`, velocità angolare entro `0,9`, `0,45 m/s` e
  `0,8 rad/s`; il controllore IsaacRobotics gestisce i giunti.
- Ogni azione dura `0,2 s`; limite episodio `80 s`. Il periodo di assestamento
  iniziale non entra nell'energia dell'episodio.
- Reward: progresso verso il goal, meno `0,005` per joule meccanico misurato,
  meno `0,01` per azione, `+25` al goal, `-12` per caduta o uscita dall'area.
  **Non** premia l'asfalto né penalizza direttamente la rampa. L'energia di
  ciascun episodio è la somma di `|tau_measured * qdot| dt` sui giunti.

La rampa e l'asfalto hanno lo stesso attrito nel USD (`statico=1,0`,
`dinamico=0,8`); cambia la geometria. La deviazione è più lunga. Perciò non
è ancora dimostrato che consumi **meno energia totale**: dipende anche dalle
curve e dalla velocità. `compare_fixed_routes.py` misura due traiettorie
guidate alla stessa velocità massima, come riferimento pratico, senza
allenare una policy. Non è una ricerca dell'ottimo assoluto.

## Comandi, dalla cartella `my-projects`

Prova tecnica breve, **senza training né checkpoint**:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/ramp_energy_choice/train.py --headless --num-envs 32 --probe-steps 8
```

Confronto opzionale della spesa energetica dei due percorsi specificati:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/ramp_energy_choice/compare_fixed_routes.py --headless --repeats 3 --speed 0.8
```

Training con 32 Spot, DINO live e 102.400 transizioni totali:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/ramp_energy_choice/train.py --headless --num-envs 32 --timesteps 102400
```

Ogni training crea una cartella nuova sotto `ramp_energy_choice/runs/`, con
`config.json`, `episodes.csv`, `timing.csv`, checkpoint intermedi ogni 5.120
transizioni e `policy_final.zip` al termine. La prova tecnica non crea un run.
Se interrompi con `Ctrl+C`, attendi la riga del checkpoint prima di chiudere
il terminale.

Dopo il training, sostituisci `<RUN>` col nome della cartella appena creata.
Senza `--headless` la valutazione apre la finestra e salva un CSV nuovo in
`ramp_energy_choice/results/`:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/ramp_energy_choice/play.py --checkpoint spot_rl_energy_navigation/ramp_energy_choice/runs/<RUN>/policy_final.zip --episodes 20
```

`ramp_entry` e `asphalt_entry` indicano che almeno un piede è passato sopra
l'area XY corrispondente; possono valere entrambi. Per confrontare l'energia
tra percorsi, considera prima di tutto gli episodi che hanno raggiunto il
goal. Un training con un'unica coppia start-goal può imparare una traiettoria
fissa: la scelta delle feature va verificata in un secondo esperimento con
scene o posizioni diverse.
