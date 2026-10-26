# Verifica offline actor/critic — 26 ottobre 2026

Analisi dei log, delle copie del codice dei run e del checkpoint finale a 350016. Nessuna simulazione o modifica dei training.

| Transizioni | Stop anticipati / rollout del run | Aggiornamenti ultimi 10 rollout / massimo | Percentuale |
|---|---|---|---|
| 200000 | 29/49 | 555/1580 | 35.1% |
| 300000 | 25/25 | 282/1510 | 18.7% |
| 350016 | 13/13 | 264/1480 | 17.8% |

## Evidenza

- Il codice salvato nei tre run coincide con ppo.py/networks.py correnti.
- In PPO.update lo stop KL interrompe il ciclo comune actor/critic prima del calcolo della loss del critic per quel minibatch. La loss e il clipping della norma dei gradienti sono congiunti, pur con reti distinte.
- Nel checkpoint finale tutti i contatori Adam sono 6286, uguali agli aggiornamenti registrati: non esistono aggiornamenti aggiuntivi del critic.
- Il massimo tiene conto dei rollout finali parziali. Gli aggiornamenti sono riusi dei dati raccolti, non transizioni aggiuntive.

## Limiti

Lo stop anticipato PPO è previsto e la percentuale di aggiornamenti non è una misura della qualità. I checkpoint non contengono i rollout (osservazioni, target GAE e gradienti), e i log non includono explained variance o norme dei gradienti separate. Non è quindi possibile provare da questi dati che il critic sia sottoallenato, che domini il clipping o che causi le traiettorie errate. MSE elevata in unità di reward non basta a stabilirlo.

## Proposta test 3

Actor e critic con parametri, ottimizzatori e clipping separati. Stop KL solo actor; critic fino a 10 epoche sui target fissati del rollout. Mantenere inizialmente learning rate 3e-4 e soglia KL 0.02; max grad norm 0.5 separatamente. Registrare numero di aggiornamenti per rete, motivo dello stop, KL che attiva lo stop, norme pre-clipping, MSE e explained variance per le due teste prima/dopo gli aggiornamenti. Explained variance nulla/non definita se target costante va segnalata, non interpretata come miglioramento. Nessuna modifica proposta al peso relativo dei due advantage.

Questa è una modifica motivata al prossimo esperimento, non una diagnosi causale dimostrata. Con rete e ottimizzazione entrambe modificate, un eventuale miglioramento non identifica separatamente i due contributi.
