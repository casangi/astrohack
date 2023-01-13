class LinearAxis:
    # According to JWS this class is superseded by xarray, which
    # should be used instead
    def __init__(self, n, ref, val, inc):
        """
        Args:
            n:   Axis size
            ref: Refence element in the axis
            val: Value at ref
            inc: Increment between axis elements
        """
        self.n = n
        self.ref = ref
        self.val = val
        self.inc = inc

    def idx_to_coor(self, idx):
        """
        Converts from an index position to a coordinate
        Args:
            idx: index position

        Returns:
        Coordinate at idx
        """
        return (idx - self.ref) * self.inc + self.val

    def coor_to_idx(self, coor):
        """
        Converts from a coordinate to an index
        Args:
            coor: coordinate position

        Returns:
        index at coor
        """
        return (coor - self.val) / self.inc + self.ref
