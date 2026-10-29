# Sensibilità al goal — SAC

Osservazioni reali del replay di training, NON dei nove episodi di test.
Il goal XY cambia direzione conservando la distanza; stessi campioni in tutte le condizioni.
Condizioni: original = stato originale; yaw_rate_zero = sola velocità angolare Z azzerata;
previous_yaw_action_zero = solo comando precedente di rotazione azzerato; both_zero = entrambi.
Azzeramenti sugli input grezzi, prima della normalizzazione. Mappa e altri input invariati.
Non viene imposto alcun limite al comando prodotto dalla rete.
Policy deterministica, normalizzazione congelata, nessuna simulazione o aggiornamento.

Campioni: 512 su 60447 disponibili entro 0.55–2.5 m; seed 42.

| Condizione | Goal (gradi; positivo=sinistra) | vx medio (m/s) | wz medio (rad/s) | Sinistra >0.1 rad/s | Delta wz vs stato originale, stesso goal |
|---|---|---:|---:|---:|---:|
| original | original | 0.718 | 1.229 | 95.7% | +0.000 |
| original | -150 | 0.292 | 1.493 | 100.0% | +0.000 |
| original | -110 | 0.269 | 1.451 | 100.0% | +0.000 |
| original | -90 | 0.293 | 1.196 | 99.6% | +0.000 |
| original | -60 | 0.489 | 0.590 | 72.3% | +0.000 |
| original | -30 | 0.864 | 0.486 | 68.2% | +0.000 |
| original | -10 | 1.023 | 0.853 | 84.8% | +0.000 |
| original | 0 | 1.064 | 1.119 | 94.1% | +0.000 |
| original | 10 | 1.083 | 1.321 | 99.6% | +0.000 |
| original | 30 | 1.063 | 1.455 | 100.0% | +0.000 |
| original | 60 | 0.973 | 1.487 | 100.0% | +0.000 |
| original | 90 | 0.828 | 1.491 | 100.0% | +0.000 |
| original | 110 | 0.701 | 1.492 | 100.0% | +0.000 |
| original | 150 | 0.452 | 1.492 | 100.0% | +0.000 |
| original | 180 | 0.337 | 1.493 | 100.0% | +0.000 |
| yaw_rate_zero | original | 0.758 | 1.255 | 96.3% | +0.026 |
| yaw_rate_zero | -150 | 0.329 | 1.493 | 100.0% | +0.000 |
| yaw_rate_zero | -110 | 0.314 | 1.469 | 100.0% | +0.018 |
| yaw_rate_zero | -90 | 0.345 | 1.276 | 99.6% | +0.079 |
| yaw_rate_zero | -60 | 0.569 | 0.732 | 79.5% | +0.143 |
| yaw_rate_zero | -30 | 0.928 | 0.659 | 75.2% | +0.173 |
| yaw_rate_zero | -10 | 1.055 | 1.042 | 92.6% | +0.189 |
| yaw_rate_zero | 0 | 1.086 | 1.260 | 97.5% | +0.142 |
| yaw_rate_zero | 10 | 1.097 | 1.386 | 99.4% | +0.066 |
| yaw_rate_zero | 30 | 1.075 | 1.465 | 100.0% | +0.010 |
| yaw_rate_zero | 60 | 1.000 | 1.487 | 100.0% | +0.000 |
| yaw_rate_zero | 90 | 0.866 | 1.492 | 100.0% | +0.000 |
| yaw_rate_zero | 110 | 0.743 | 1.492 | 100.0% | +0.000 |
| yaw_rate_zero | 150 | 0.491 | 1.492 | 100.0% | +0.000 |
| yaw_rate_zero | 180 | 0.371 | 1.493 | 100.0% | +0.000 |
| previous_yaw_action_zero | original | 0.700 | 1.240 | 96.1% | +0.011 |
| previous_yaw_action_zero | -150 | 0.283 | 1.493 | 100.0% | +0.000 |
| previous_yaw_action_zero | -110 | 0.264 | 1.468 | 100.0% | +0.016 |
| previous_yaw_action_zero | -90 | 0.287 | 1.273 | 99.6% | +0.077 |
| previous_yaw_action_zero | -60 | 0.471 | 0.695 | 77.1% | +0.106 |
| previous_yaw_action_zero | -30 | 0.853 | 0.564 | 71.5% | +0.078 |
| previous_yaw_action_zero | -10 | 1.014 | 0.919 | 87.5% | +0.066 |
| previous_yaw_action_zero | 0 | 1.054 | 1.169 | 95.9% | +0.051 |
| previous_yaw_action_zero | 10 | 1.071 | 1.347 | 99.6% | +0.027 |
| previous_yaw_action_zero | 30 | 1.047 | 1.461 | 100.0% | +0.005 |
| previous_yaw_action_zero | 60 | 0.956 | 1.488 | 100.0% | +0.001 |
| previous_yaw_action_zero | 90 | 0.812 | 1.492 | 100.0% | +0.000 |
| previous_yaw_action_zero | 110 | 0.685 | 1.492 | 100.0% | +0.000 |
| previous_yaw_action_zero | 150 | 0.437 | 1.492 | 100.0% | +0.000 |
| previous_yaw_action_zero | 180 | 0.324 | 1.493 | 100.0% | +0.000 |
| both_zero | original | 0.743 | 1.261 | 96.1% | +0.032 |
| both_zero | -150 | 0.318 | 1.493 | 100.0% | +0.000 |
| both_zero | -110 | 0.306 | 1.476 | 100.0% | +0.025 |
| both_zero | -90 | 0.334 | 1.332 | 99.8% | +0.136 |
| both_zero | -60 | 0.548 | 0.819 | 85.0% | +0.229 |
| both_zero | -30 | 0.918 | 0.719 | 79.9% | +0.233 |
| both_zero | -10 | 1.047 | 1.085 | 94.1% | +0.232 |
| both_zero | 0 | 1.078 | 1.284 | 97.9% | +0.165 |
| both_zero | 10 | 1.088 | 1.397 | 99.6% | +0.076 |
| both_zero | 30 | 1.063 | 1.468 | 100.0% | +0.012 |
| both_zero | 60 | 0.989 | 1.488 | 100.0% | +0.001 |
| both_zero | 90 | 0.855 | 1.492 | 100.0% | +0.000 |
| both_zero | 110 | 0.731 | 1.492 | 100.0% | +0.000 |
| both_zero | 150 | 0.478 | 1.492 | 100.0% | +0.000 |
| both_zero | 180 | 0.358 | 1.493 | 100.0% | +0.000 |

## Interpretazione e limiti

Una risposta diversa a destra e sinistra dimostra sensibilità al goal, non una navigazione corretta.
Una risposta quasi invariata suggerisce scarso uso del goal o predominanza degli altri input.
La rotazione verso il goal non è sempre la scelta corretta in presenza di ostacoli.
I goal modificati possono essere fuori dalla distribuzione di training o su zone non percorribili.
Anche azzerare velocità o azione precedente può creare stati fuori distribuzione o incoerenti con gli altri input.
Una variazione dopo gli azzeramenti isola una dipendenza funzionale, non prova che tali input vadano eliminati.
Il test isola l’effetto immediato dell’input goal sui comandi; non identifica da solo la causa di apprendimento,
non misura la dinamica futura e non dimostra un difetto della reward. I campioni temporali sono correlati.
