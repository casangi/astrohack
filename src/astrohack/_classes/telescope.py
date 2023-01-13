class Telescope:

    def __init__(self, name):
        """
        Initializes antenna surface relevant information based on the telescope name
        Args:
            name: telescope name
        """
        if name == 'VLA':
            self._init_vla()
        elif name == 'VLBA':
            self._init_vlba()
        else:
            raise Exception("Unknown telescope: " + name)
        return

    # Other known telescopes should be included here, ALMA, ngVLA
    def _init_vla(self):
        """
        Initializes object according to parameters specific to VLA panel distribution
        """
        self.name = "VLA"
        self.diam = 25.0  # meters
        self.focus = 8.8  # meters
        self.ringed = True
        self.nrings = 6
        self.npanel = [12, 16, 24, 40, 40, 40]
        self.inrad = [1.983, 3.683, 5.563, 7.391, 9.144, 10.87]
        self.ourad = [3.683, 5.563, 7.391, 9.144, 10.87, 12.5]
        self.inlim = 2.0
        self.oulim = 12.0

    def _init_vlba(self):
        """
        Initializes object according to parameters specific to VLBA panel distribution
        """
        self.name = "VLBA"
        self.diam = 25.0  # meters
        self.focus = 8.75  # meters
        self.ringed = True
        self.nrings = 6
        self.npanel = [20, 20, 40, 40, 40, 40]
        self.inrad = [1.676, 3.518, 5.423, 7.277, 9.081, 10.808]
        self.ourad = [3.518, 5.423, 7.277, 9.081, 10.808, 12.500]
        self.inlim = 2.0
        self.oulim = 12.0
