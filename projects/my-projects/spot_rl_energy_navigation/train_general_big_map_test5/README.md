# Navigazione su mappa grande — test 5

Training **da zero, 200.000 transizioni RL, 32 ambienti separati**. Deriva dal test 4: stessa mappa congelata `maps/real_terrains_v2.usd`, banca missioni e PCA32 già calibrata sui sette materiali. Restano invariati rete, PPO, osservazioni, curriculum, allineamento e limiti di velocità. Non riprende i pesi del test 4.

## Comandi

Dalla radice del progetto:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test5/train.sh
```

Monitoraggio:

```bash
watch -n 10 python3 spot_rl_energy_navigation/train_general_big_map_test5/status.py
```

Test visibile a fine training (ultimo `policy_final.pt`, nove missioni, un robot):

```bash
bash spot_rl_energy_navigation/train_general_big_map_test5/test.sh
```

Il test usa l'ultima fase valutata e il seed 20000. Accetta `--stage 0`, `--episodes 20`, oppure `--checkpoint /percorso/policy_latest.pt`. Salva i risultati in una nuova sottocartella `results/`.

Il launcher usa Isaac Lab 2.3.0, CUDA per simulazione/DINO e CPU per PPO. Mantiene il supporto di `GENERAL_MAP_ISAACLAB` e dell'eventuale `../cloud/env.sh`. Verifica e riusa `artifacts/pca32_big_map.npz`, copiata dal test 4: non occorre una nuova raccolta. Se manca, è disponibile la calibrazione automatica sui sette terreni. Ogni checkpoint incorpora la PCA utilizzata; questa rimane congelata durante RL.

Per riprendere esclusivamente un checkpoint del **test 5**:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test5/train.sh \
  --resume /percorso/policy_final.pt --timesteps 400000
```

Il budget è il totale finale desiderato. I checkpoint precedenti sono rifiutati perché hanno condizioni dell'esperimento diverse.

## Cadute con possibilità di recupero

Una caduta viene confermata se almeno una delle condizioni persiste per **1 secondo consecutivo**:

- Inclinazione della base rispetto alla verticale **maggiore o uguale a 90°**.
- Forza di contatto del corpo con il terreno **maggiore di 1 N**.

I due contatori sono indipendenti e si azzerano quando la rispettiva condizione scompare: contatti brevi separati non si sommano. Campionamento ogni **0,005 s**, quindi servono 200 campioni consecutivi. Un evento confermato rimane memorizzato fino al reset, effettuato al confine dell'azione da 0,2 s. La policy e il controllore continuano a operare durante il periodo di possibile recupero.

Il sensore usa il rigid body `Spot/body`, verificato nell'asset USD come unico corpo con collisioni che non appartenga alle gambe. Filtra esclusivamente le 49 mesh del terreno: piedi, gambe e contatti interni non entrano nel criterio. Si sommano le norme delle forze per gruppo terreno, evitando la cancellazione tra contatti opposti. La soglia di 1 N ignora forze trascurabili; è configurabile.

**Altezza della base sotto 25 cm e misura del terreno non valida non provocano più una caduta.** Rimangono diagnostiche. Un'altezza non disponibile produce `null` nei JSON e un campo vuoto nei CSV. Dati fisici non finiti di orientamento/contatto generano un errore tecnico esplicito, non un falso episodio `fallen`.

Goal, uscita dai limiti, timeout missione e timeout di allineamento restano invariati e possono terminare un episodio indipendentemente dai contatori di caduta. Non viene introdotta una pausa artificiale per il video.

## Reward

Per ogni azione di navigazione (dt = 0,2 s):

```text
r = 30 * (d_precedente - d_attuale) - 0.02
    + 1000 * successo - 1000 * fallimento
    - energia_step_J / 100
    - 2.0 * dt * clip((abs(errore_goal_gradi) - 110) / 70, 0, 1)
```

La penalità di orientamento resta zero fino a 110° e raggiunge **−2 punti/s a 180°**, cioè −0,4 per passo. Usa la posa dopo il passo ed è esclusa durante l'allineamento automatico iniziale. È inclusa nella testa navigazione del critic; la testa energetica è invariata. L'energia integra `sum(abs(tau*qdot))` su tutti i passi fisici.

## Diagnostica

`episodes.csv` e i JSON delle valutazioni includono:

- `fall_cause`: `tilt`, `body_contact`, `tilt+body_contact`, oppure vuoto.
- Inclinazione e forza di contatto finali e massime dell'episodio.
- Durate consecutive correnti e massime dei due segnali, misurate ai passi fisici.
- Validità della misura del terreno e distanza finale base-terreno.
- `low_clearance_steps`, `invalid_height_steps`, `legacy_fall_steps`: numero di passi **da 0,2 s** che avrebbero attivato i segnali del vecchio criterio; non causano reset.

Il test salva anche **`fall_diagnostics.csv`**, con una riga per passo da 0,2 s, episodio, tempo, XYZ e diagnostica. I dati sono acquisiti **dopo il passo e prima del reset**, incluso il campione terminale. `trajectories.csv` mantiene invece i campioni pre-step: parte da step 0, mentre la diagnostica parte da step 1. Il terminale stampa motivo e massimi dell'evento a fine episodio.

La nuova diagnostica non aggiunge input alla rete. I dati e le policy dei test precedenti restano invariati.

## Configurazione conservata

- Actor: stato 15→64→32; mappa 3240→1024→512→256→128; fusione 160→128→64→2. Critic separato con 24 informazioni privilegiate aggiuntive e due teste.
- Gridmap robocentrica 9×9, passo 0,5 m, PCA32 e geometria/validità/età; memoria separata per episodio.
- Comandi vx∈[0,1.6] m/s e wz∈[−1.5,1.5] rad/s.
- Allineamento fisico automatico: tre misure consecutive entro 10°, limite 12 s. Nessun riallineamento durante la navigazione.
- Timeout `clip(15 + 4*d_iniziale, 30, 90)` s, incluso allineamento. Raggio goal 0,55 m.
- Curriculum 0–4: stessa cella asfalto 1,5–3 m; terreni misti 3–6 m senza endpoint su scale/ostacoli; tutti gli endpoint ammessi 3–6 m; 6–12 m; 12–20 m. Ghiaccio escluso dagli spawn, ghiaccio e scale esclusi dai goal. Gli altri terreni rimangono nella mappa.
- Promozione invariata: 80% in due valutazioni consecutive, 32 missioni deterministiche per fase. Ripasso con probabilità 25%.
- PPO: circa 4096 transizioni per rollout, batch 256, 10 epoche, learning rate 3e-4, gamma 0,999, lambda 0,95, clip 0,2, entropia 0,005. Ottimizzatori separati, stop actor KL 0,02.
- Il budget conta solo campioni RL. Allineamento escluso dal buffer PPO, ma incluso nei totali delle missioni. Valutazioni periodiche ogni circa 10.000 transizioni più quella finale.

## Verifiche

Test offline su persistenza, recuperi transitori, reset parziali, contatti filtrati, altezza mancante/bassa senza terminazione, diagnostica JSON e reward aggiornata, oltre ai test ereditati di PPO, percezione, curriculum e launcher. Verificati asset, firma dell'esperimento e PCA. Nessun training o simulazione del test 5 avviato durante la preparazione: il nuovo sensore nel motore fisico deve essere verificato al primo avvio.
