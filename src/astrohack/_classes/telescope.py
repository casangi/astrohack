ring_list_keys = ['npanel', 'inrad', 'ourad']
ring_simple_keys = ['name', 'diam', 'focus', 'nrings', 'inlim', 'oulim']


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

    def _read_cfg_file(self, filename):
        try:
            cfgfile = open(filename, 'r')
            ledict = {}
            for line in cfgfile:
                wrds = line.split('=')
                ledict[wrds[0].strip()] = wrds[1]
            cfgfile.close()
        except:
            raise Exception('Badly formatted cfg file')

        print(ledict)
        print(ledict['ringed'])
        try:
            self.ringed = bool(ledict['ringed'])
        except KeyError:
            raise Exception('Ringed keyword missing from cfg file')
        except ValueError:
            raise Exception('Value for keyword ringed is not boolean')

        if self.ringed:
            self._init_ringed_telescope(ledict)
        else:
            self._init_general_telescope(ledict)

        return

    def _init_ringed_telescope(self, ledict):
        for key in ring_simple_keys:
            try:
                setattr(self, key, float(ledict[key]))
            except KeyError:
                raise Exception(key+' keyword missing from cfg file')
            except ValueError:
                raise Exception('Cannot convert '+ledict[key]+' to float')

        for key in ring_list_keys:
            try:
                wrds = ledict[key].split(',')
                lelist = []
                for word in wrds:
                    lelist.append(float(word))
                setattr(self, key, lelist)
            except KeyError:
                raise Exception(key+' keyword missing from cfg file')
            except ValueError:
                raise Exception('Failed to convert values to float for keyword '+key)

        if not self.npanel == len(self.npanel) == len(self.inrad) == self.ourad:
            raise Exception('Number of panels don\'t match radii or number of panels list sizes')

        return

    def _init_general_telescope(self, ledict):
        raise Exception("General layout telescopes not yet supported")
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
