import configparser


class PytchConfig:
    device_index = None
    accept = True
    show_traces = True
    start_full_screen = False

    def set_menu(self, m):
        if isinstance(m, ProcessingMenu):
            m.box_show_traces.setChecked(self.show_traces)

    def dump(self, filename):
        pass


