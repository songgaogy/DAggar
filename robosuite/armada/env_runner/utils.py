from pynput import keyboard as pynput_keyboard


class AsyncKeyHandler:
    """
    Real asynchronous keyboard listener using pynput.
    Listens for global control keys independent of the simulation window focus.
    """
    def __init__(self):
        self.finish = False
        self.help = False
        self.discard = False
        self.back = False
        self.pause = False
        self.ctn = False
        self.rst = False
        
        # Start listener in a non-blocking way
        self.listener = pynput_keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        print("\n[INFO] Async Keyboard Listener started. Hotkeys: [C]ontinue, [H]elp, [D]iscard, [F]inish, [B]ack, [P]ause, [R]eset")
        print("    - [C]ontinue: continue rollout")
        print("    - [H]elp: human intervention")
        print("    - [D]iscard: Discard all collected data since very beginning")
        print("    - [F]inish: quit and save collected data")
        print("    - [B]ack: end human intervention")
        print("    - [R]eset: reset the env and start again")
        print("    - [P]ause: pause current env, press [P] again to continue\n")

    def on_press(self, key):
        try:
            # Check for char keys
            if hasattr(key, 'char') and key.char is not None:
                k = key.char.lower()
                if k == 'h': self.help = True
                elif k == 'd': self.discard = True
                elif k == 'f': self.finish = True
                elif k == 'b': self.back = True
                elif k == 'p': self.pause = True
                elif k == 'c': self.ctn = True
                elif k == 'r': self.rstn = True
        except AttributeError:
            pass

    def reset(self):
        self.finish = False
        self.help = False
        self.discard = False
        self.ctn = False
        self.pause = False
        self.back = False
        self.rst = False
    
    def stop(self):
        if self.listener:
            self.listener.stop()