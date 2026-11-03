# Test 8 SAC — tutti gli scaler degli input congelati

Training da zero, **600.000 transizioni totali**, 32 ambienti separati. Stessa
mappa, PCA, missioni, curriculum, reward, rete, controllore e impostazioni SAC
predefinite del test 7. Nessun limite nuovo alle rotazioni.

## Avvio

Dalla directory principale del progetto:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test8_SAC/train.sh
```

Il comando verifica e riusa la PCA e gli scaler già preparati, poi avvia il training.
Non serve una nuova calibrazione in Isaac Sim. Non usa pesi di policy precedenti.
Per monitorare:

```bash
watch -n 10 python3 spot_rl_energy_navigation/train_general_big_map_test8_SAC/status.py
```

Al termine:

```bash
bash spot_rl_energy_navigation/train_general_big_map_test8_SAC/test.sh
```

Test visibile: ultimo modello finale, nove missioni, seed 20000, ultima fase valutata.
Per un confronto esplicito sulla fase 1 aggiungere `--stage 1`.

## Trasformazioni fisse

| Input | Trasformazione |
|---|---|
| Goal XY | /10 m; nessuna media, nessun clipping |
| Velocità XYZ | /2 m/s |
| Velocità angolari XYZ | /2 rad/s |
| Potenza meccanica EMA | /400 W |
| Altezza relativa | /0,7 m |
| Pendenze XY (radianti) | /(π/2) |
| Rugosità | /0,05 m |
| Feature PCA32 | (x − media congelata) / deviazione standard congelata |
| Attrito statico/dinamico (critic) | Scala 1, senza centratura |
| Forze delle quattro zampe (critic) | /300 N |
| Scorrimento XY delle zampe (critic) | /1,5 m/s |

Rimane il clipping a ±10 per i gruppi trasformati, tranne il goal. Restano invariati
gravità proiettata, azioni precedenti, maschere, tempo residuo ed età delle celle.
Le feature delle celle e i privilegi senza contatto vengono mascherati DOPO la
trasformazione; le statistiche visive sono condivise tra tutte le celle.

Non ci sono aggiornamenti degli scaler né alla raccolta né negli update SAC.
Gli stessi buffer sono usati da actor, critic, target input, valutazione ed export.
Il metodo `normalizer.update` è mantenuto solo per compatibilità ed è un no-op.

## Calibrazione già eseguita offline

Artefatto: `artifacts/frozen_scalers.json`. Il suo hash è parte della firma dei
checkpoint; una copia viene conservata in `runs/.../sources`. Non riprendere
checkpoint dei test 6/7 con questo esperimento.

- Visivo: 70.000 patch DINO, 10.000 per ciascuno dei sette materiali, dal dataset
  di calibrazione del test 5. La PCA del test 5 e quella del test 7 sono identiche
  byte per byte. Si proiettano le patch con questa PCA senza riaddestrarla.
- Le statistiche vengono calcolate solo sulle patch del set di calibrazione,
  non sulle viste di validazione. La deviazione standard ha un minimo esplicito
  per non amplificare componenti quasi costanti.
- Fisico: campioni riproducibili (4096 per replay, seed 123) dalle esperienze di
  training del test 6 a 300k e del test 7 a 200k. Si usa il percentile 99 dei valori
  assoluti, arrotondato verso l'alto, con riferimenti minimi fisici. Geometria:
  sole celle valide. Forze e scorrimento: soli contatti validi.
- Non sono state usate traiettorie delle valutazioni. Non vengono importati i pesi
  degli actor o dei critic precedenti.

Limiti: il replay dinamico non è bilanciato su tutti i terreni; i divisori fisici
sono riferimenti operativi, non limiti massimi garantiti. Le statistiche visive
provengono da patch prima della media sulle celle: la variabilità delle celle
aggregate può essere minore. Queste scelte sono fisse e documentate, non adattate
ai risultati del nuovo training. La sola sostituzione degli scaler non garantisce
che il modello apprenda meglio.

`prepare_scalers.py` riproduce la calibrazione usando i file sorgenti precedenti;
rifiuta di sovrascrivere un artefatto esistente. Una ricalibrazione va trattata
come nuovo esperimento. Il training ordinario non dipende dai replay precedenti,
ma soltanto dall'artefatto già incluso e dagli asset condivisi.

## Verifiche offline

```bash
PYTHONPATH=spot_rl_energy_navigation OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -m pytest -q spot_rl_energy_navigation/train_general_big_map_test8_SAC
```

Copertura: buffer immutabili con dati di distribuzione diversa, unità fisiche,
maschere, actor/critic, checkpoint/export, raccolta replay senza aggiornamento
statistiche, orchestrazione SAC con ambiente fittizio, mappe CPU/GPU.
