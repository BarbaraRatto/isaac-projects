# Sensibilità al goal — SAC

Osservazioni reali del replay di training, NON dei nove episodi di test.
Intervento: cambia soltanto il goal XY robocentrico, conservandone la distanza.
Mappa, velocità, azione precedente e tutti gli altri input restano invariati.
Policy deterministica, normalizzazione congelata, nessuna simulazione o aggiornamento.

Campioni: 512 su 60447 disponibili entro 0.55–2.5 m; seed 42.

| Goal (gradi; positivo=sinistra) | vx medio (m/s) | wz medio (rad/s) | Sinistra >0.1 rad/s |
|---|---:|---:|---:|
| original | 0.718 | 1.229 | 95.7% |
| -150 | 0.292 | 1.493 | 100.0% |
| -110 | 0.269 | 1.451 | 100.0% |
| -90 | 0.293 | 1.196 | 99.6% |
| -60 | 0.489 | 0.590 | 72.3% |
| -30 | 0.864 | 0.486 | 68.2% |
| -10 | 1.023 | 0.853 | 84.8% |
| 0 | 1.064 | 1.119 | 94.1% |
| 10 | 1.083 | 1.321 | 99.6% |
| 30 | 1.063 | 1.455 | 100.0% |
| 60 | 0.973 | 1.487 | 100.0% |
| 90 | 0.828 | 1.491 | 100.0% |
| 110 | 0.701 | 1.492 | 100.0% |
| 150 | 0.452 | 1.492 | 100.0% |
| 180 | 0.337 | 1.493 | 100.0% |

## Interpretazione e limiti

Una risposta diversa a destra e sinistra dimostra sensibilità al goal, non una navigazione corretta.
Una risposta quasi invariata suggerisce scarso uso del goal o predominanza degli altri input.
La rotazione verso il goal non è sempre la scelta corretta in presenza di ostacoli.
I goal modificati possono essere fuori dalla distribuzione di training o su zone non percorribili.
Il test isola l’effetto immediato dell’input goal sui comandi; non identifica da solo la causa di apprendimento,
non misura la dinamica futura e non dimostra un difetto della reward. I campioni temporali sono correlati.
