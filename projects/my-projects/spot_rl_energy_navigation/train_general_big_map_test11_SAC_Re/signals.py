"""Defer terminal signals until a complete vector-step boundary, including evaluation."""
import signal


class StopRequest:
    def __init__(self):
        self.requested = False
        self.originals = {}

    def receive(self, signum, frame):
        if not self.requested:
            print('[SAC STOP] Richiesta ricevuta: termino il passo corrente e salvo checkpoint + replay. '
                  'Attendi il messaggio Saved e il ritorno del prompt.', flush=True)
        self.requested = True

    def install(self):
        # Install AFTER AppLauncher, which may register its own signal handlers.
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.originals[sig] = signal.getsignal(sig)
            signal.signal(sig, self.receive)

    def restore(self):
        for sig, handler in self.originals.items():
            signal.signal(sig, handler)
        self.originals.clear()
