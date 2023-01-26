import numpy as np
from scipy import optimize as opt


def _gauss_elimination_numpy(system, vector):
    """
    Gauss elimination solving of a system using numpy
    Args:
        system: System matrix to be solved
        vector: Vector that represents the right hand side of the system

    Returns:
    The solved system
    """
    inverse = np.linalg.inv(system)
    return np.dot(inverse, vector)


class BasePanel:
    markers = ['X', 'o', '*', 'P', 'D']
    colors = ['g', 'g', 'r', 'r', 'b']
    fontsize = 5
    linewidth = 0.5
    markersize = 2
    linecolor = 'black'

    def __init__(self, kind, ipanel, screws):
        """
        Initializes a BasePanel with the common machinery to both PolygonPanel and RingPanel
        Args:
            kind: What kind of surface to be used in fitting ["rigid", "mean", "xyparaboloid",
            "rotatedparaboloid", "corotatedparaboloid"]
            ipanel: Panel numbering
            screws: position of the screws
        """
        self.kind = kind
        self.solved = False
        self.ipanel = ipanel + 1
        self.screws = screws
        self.nsamp = 0
        self.values = []
        self.corr = None

        # These are overridden by the derived classes
        self.center = [0, 0]
        self.zeta = 0

        if self.kind == "rigid":
            self.solve = self._solve_rigid
            self.corr_point = self._corr_point_rigid
        elif self.kind == "mean":
            self.solve = self._solve_mean
            self.corr_point = self._corr_point_mean
        elif self.kind == "xyparaboloid":
            self.solve = self._solve_scipy
            self.corr_point = self._corr_point_scipy
            self._paraboloid = self._xyaxes_paraboloid
        elif self.kind == "rotatedparaboloid":
            self.solve = self._solve_scipy
            self.corr_point = self._corr_point_scipy
            self._paraboloid = self._rotated_paraboloid
        elif self.kind == "corotatedparaboloid":
            self.solve = self._solve_scipy
            self.corr_point = self._corr_point_scipy
            self._paraboloid = self._corotated_paraboloid
        else:
            raise Exception("Unknown panel kind: ", self.kind)

    def add_point(self, value):
        """
        Add a point to the panel's list of point to be fitted
        Args:
            value: tuple/list containing point description [xcoor,ycoor,xidx,yidx,value]
        """
        self.values.append(value)
        self.nsamp += 1

    def _solve_scipy(self, verbose=False):
        """
        Fit ponel surface by using arbitrary models using scipy surface fitting engine
        Args:
            verbose: Increase verbosity in the fitting process
        """
        devia = np.ndarray([self.nsamp])
        coords = np.ndarray([2, self.nsamp])
        for i in range(self.nsamp):
            devia[i] = self.values[i][-1]
            coords[:, i] = self.values[i][0], self.values[i][1]

        liminf = [0, 0, -np.inf]
        limsup = [np.inf, np.inf, np.inf]
        p0 = [1e2, 1e2, np.mean(devia)]

        if self.kind == "thetaparaboloid":
            liminf.append(0.0)
            limsup.append(np.pi)
            p0.append(0)

        maxfevs = [100000, 1000000, 10000000]
        for maxfev in maxfevs:
            try:
                result = opt.curve_fit(self._paraboloid, coords, devia,
                                       p0=p0, bounds=[liminf, limsup],
                                       maxfev=maxfev)
            except RuntimeError:
                if verbose:
                    print("Increasing number of iterations")
                continue
            else:
                self.par = result[0]
                self.solved = True
                if verbose:
                    print("Converged with less than {0:d} iterations".format(maxfev))
                break

    def _xyaxes_paraboloid(self, coords, xcurv, ycurv, zoff):
        """
        Surface model to be used in fitting with scipy
        Assumes that the center of the paraboloid is the center of the panel
        In this model the panel can only bend in the x and y directions
        Args:
            coords: [x,y] coordinate pair for point
            xcurv: curvature in x direction
            ycurv: curvature in y direction
            zoff:  Z offset of the paraboloid

        Returns:
        Paraboloid value at X and Y
        """
        x, y = coords
        return -(((x - self.center[0]) / xcurv) ** 2 + ((y - self.center[1]) / ycurv) ** 2) + zoff

    def _rotated_paraboloid(self, coords, ucurv, vcurv, zoff, theta):
        """
        Surface model to be used in fitting with scipy
        Assumes that the center of the paraboloid is the center of the panel
        This model is degenerate in the combinations of theta, ucurv and vcurv
        Args:
            coords: [x,y] coordinate pair for point
            ucurv: curvature in projected u direction
            vcurv: curvature in projected v direction
            zoff:  Z offset of the paraboloid
            theta: Angle between x,y and u,v coordinate systems

        Returns:
        Paraboloid value at X and Y
        """
        x, y = coords
        xc, yc = self.center
        u = (x - xc) * np.cos(theta) + (y - yc) * np.sin(theta)
        v = (x - xc) * np.sin(theta) + (y - yc) * np.cos(theta)
        return -((u / ucurv) ** 2 + (v / vcurv) ** 2) + zoff

    def _corotated_paraboloid(self, coords, ucurv, vcurv, zoff):
        """
        Surface model to be used in fitting with scipy
        Same as the rotated paraboloid above, but theta is the panel center angle
        Not valid for polygon panels
        Assumes that the center of the paraboloid is the center of the panel
        Args:
            coords: [x,y] coordinate pair for point
            ucurv: curvature in projected u direction
            vcurv: curvature in projected v direction
            zoff:  Z offset of the paraboloid

        Returns:
        Paraboloid value at X and Y
        """
        x, y = coords
        xc, yc = self.center
        u = (x - xc) * np.cos(self.zeta) + (y - yc) * np.sin(self.zeta)
        v = (x - xc) * np.sin(self.zeta) + (y - yc) * np.cos(self.zeta)
        return -((u / ucurv) ** 2 + (v / vcurv) ** 2) + zoff

    def _solve_rigid(self):
        """
        Fit panel surface using AIPS gaussian elimination model for rigid panels
        """
        syssize = 3
        if self.nsamp < syssize:
            # In this case the matrix will always be singular as the
            # rows will be linear combinations
            return
        system = np.zeros([syssize, syssize])
        vector = np.zeros(syssize)
        for ipoint in range(len(self.values)):
            if self.values[ipoint][-1] != 0:
                system[0, 0] += self.values[ipoint][0] * self.values[ipoint][0]
                system[0, 1] += self.values[ipoint][0] * self.values[ipoint][1]
                system[0, 2] += self.values[ipoint][0]
                system[1, 0] = system[0, 1]
                system[1, 1] += self.values[ipoint][1] * self.values[ipoint][1]
                system[1, 2] += self.values[ipoint][1]
                system[2, 0] = system[0, 2]
                system[2, 1] = system[1, 2]
                system[2, 2] += 1.0
                vector[0] += self.values[ipoint][-1] * self.values[ipoint][0]
                vector[1] += self.values[ipoint][-1] * self.values[ipoint][1]
                vector[2] += self.values[ipoint][-1]

        self.par = _gauss_elimination_numpy(system, vector)
        self.solved = True
        return

    def _solve_mean(self):
        """
        Fit panel surface as a simple mean of its points Z deviation
        """
        if self.nsamp > 0:
            # Solve panel adjustments for rigid vertical shift only panels
            self.par = np.zeros(1)
            shiftmean = 0.
            ncount = 0
            for value in self.values:
                if value[-1] != 0:
                    shiftmean += value[-1]
                    ncount += 1

            shiftmean /= ncount
            self.par[0] = shiftmean
            self.solved = True
        else:
            self.solved = False
        return

    def get_corrections(self):
        """
        Store corrections for the fitted panel points
        """
        if not self.solved:
            raise Exception("Cannot correct a panel that is not solved")
        self.corr = np.ndarray([len(self.values)])
        icorr = 0
        for val in self.values:
            self.corr[icorr] = self.corr_point(val[0], val[1])
            icorr += 1

    def _corr_point_scipy(self, xcoor, ycoor):
        """
        Computes fitted value for point [xcoor, ycoor] using the scipy models
        Args:
            xcoor: X coordinate of point
            ycoor: Y coordinate of point

        Returns:
        Fitted value at xcoor,ycoor
        """
        corrval = self._paraboloid([xcoor, ycoor], *self.par)
        return corrval

    def _corr_point_rigid(self, xcoor, ycoor):
        """
        Computes fitted value for point [xcoor, ycoor] using AIPS gaussian elimination model for rigid panels
        Args:
            xcoor: X coordinate of point
            ycoor: Y coordinate of point

        Returns:
        Fitted value at xcoor,ycoor
        """
        return xcoor * self.par[0] + ycoor * self.par[1] + self.par[2]

    def _corr_point_mean(self, xcoor, ycoor):
        """
        Computes fitted value for point [xcoor, ycoor] using AIPS shift only panels
        Args:
            xcoor: X coordinate of point
            ycoor: Y coordinate of point

        Returns:
        Fitted value at xcoor,ycoor
        """
        return self.par[0]

    def export_screw_adjustments(self, unit='mm'):
        """
        Exports panel screw adjustments to a string
        Args:
            unit: Unit for screw adjustments ['mm','miliinches']
        Returns:
        String with screw adjustments for this panel
        """
        if unit == 'mm':
            fac = 1.0
        elif unit == 'miliinches':
            fac = 1000.0 / 25.4
        else:
            raise Exception("Unknown unit: " + unit)

        string = ''
        for screw in self.screws[:, ]:
            string += ' {0:10.2f}'.format(fac * self.corr_point(*screw))
        return string

    def plot_label(self, ax):
        """
        Plots panel label to ax
        Args:
            ax: matplotlib axes instance
        """
        ax.text(self.center[0], self.center[1], str(self.ipanel), fontsize=self.fontsize, ha='center')

    def plot_screws(self, ax):
        """
        Plots panel screws to ax
        Args:
            ax: matplotlib axes instance
        """
        for iscrew in range(self.screws.shape[0]):
            screw = self.screws[iscrew, ]
            ax.scatter(screw[0], screw[1], marker=self.markers[iscrew], lw=self.linewidth, s=self.markersize,
                       color=self.colors[iscrew])
