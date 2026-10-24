# Prova manuale asfalto / rocce

Da `my-projects`, avviare (GUI, nessun training):

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/manual_terrain.py --device cuda:0 --speed 0.9
```

La mappa predefinita è `terrain_generator/manual_asphalt_rocks/rocks_right.usd`.
Stessa disposizione 3x3 del test 10: partenza (0,0), arrivo di riferimento
(4,4), ghiaccio al centro, rocce a destra e asfalto sul percorso sinistro.
Non c'è terminazione automatica al goal. Per la speculare aggiungere:

```text
--terrain terrain_generator/manual_asphalt_rocks/rocks_left.usd
```

Fare clic sulla viewport. W/S o frecce su/giù comandano avanzamento/retromarcia;
A/D o frecce sinistra/destra comandano rotazione. Rilasciare i tasti ferma
il comando. R riporta Spot alla partenza, ESC esce. Dopo il reset attendere
`[MANUAL] Ready.` (2 secondi simulati). Non usare Stop/Play per resettare.

SPAZIO avvia una nuova misura; premuto di nuovo la termina e stampa il risultato.
La misura integra `sum(abs(tau_physx * joint_velocity))` a 200 Hz con trapezi,
riporta joule, distanza percorsa e J/m nel terminale; non salva un CSV.
Include soste e rotazioni fra le due pressioni. Esclude il primo intervallo
per inizializzare il campione di potenza. I tempi sono simulati, non reali.
Per confrontare terreni usare stessa velocità, lunghezza e manovre, con più
ripetizioni; il confronto manuale è esplorativo. Il controller locomotorio
è quello congelato dei benchmark; la policy di navigazione non interviene.

Il mondo completo esistente è accessibile con:

```bash
/home/isaac/isaaclab/2.3.0/isaaclab.sh -p spot_rl_energy_navigation/manual_terrain.py --device cuda:0 --terrain terrain_generator/real_terrains.usd --spawn -1.3 8.25 0.75
```

## Modifica delle rocce

In `terrain_generator/create_real_terrains.py`, blocco `t5_large_rocks`,
`noise_range=(0.02, 0.05)` definisce le quote campionate in metri. Per esempio
`(0.02, 0.08)` aumenta l'escursione da 3 a 6 cm. È un esperimento da verificare,
non una garanzia di maggiore consumo. `noise_step=0.001` è la discretizzazione
verticale, non la dimensione delle rocce. `downsampled_scale` (facoltativo nel
sub_cfg) controlla la spaziatura orizzontale del rumore; il default qui eredita
`horizontal_scale=0.02`. Modificare un parametro alla volta.
`roughness=0.85` è solo un parametro del materiale visivo.

Le modifiche Python hanno effetto solo dopo rigenerazione degli USD. Lo script
originale sovrascrive `real_terrains.usd` e rimescola la scacchiera casualmente:
per conservare il riferimento, copiare lo script in `terrain_generator/`,
modificarne sia i parametri sia `usd_path` finale verso un nuovo file
`real_terrains_rougher.usd`, ed eseguire quella copia con `isaaclab.sh -p`.
Poi ricreare le sole mappe manuali usando il nuovo file:

```bash
/home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh terrain_generator/create_manual_asphalt_rocks.py --source terrain_generator/real_terrains_rougher.usd --overwrite
```

Chiudere e riaprire la scena manuale per caricare il nuovo terreno.
Le mappe già esportate del training non cambiano automaticamente.
Il generatore delle mappe manuali usa solo USD Python e non avvia la simulazione.
