# Test13 SAC Re

Pacchetto unico `spot-test13-sac-re-cloud.zip`: nuovo training grigio/ghiaccio nero, 128 robot, 1.500.000 transizioni, 16 aggiornamenti SAC.

La guida operativa riassuntiva completa è [COMANDI_TEST13_RE.md](COMANDI_TEST13_RE.md), inclusa alla radice dell'archivio come `LEGGIMI_TEST13_RE.md`: trasferimento, setup, tmux, training, monitor, recovery, test da 9 episodi, grafici automatici e backup.

Le proprietà del training sono descritte in [README del test13](../train_general_big_map_test13_SAC_Re/README.md). Nessuna penalità esplicita sul materiale: il nero usa la fisica del vecchio ghiaccio, attrito statico 0,10 e dinamico 0,08, senza resistenza aggiuntiva. Start e goal solo sul grigio a ogni livello.
