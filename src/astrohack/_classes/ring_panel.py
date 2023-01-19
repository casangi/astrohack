import numpy as np
from matplotlib import pyplot as plt
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


class RingPanel:
    # This class describes and treats panels that are arranged in
    # rings on the Antenna surface

    def __init__(self, kind, angle, iring, ipanel, inrad, ourad):
        """
        Initializes a panel that is a section of a ring in a circular antenna
        Args:
            kind: What kind of surface to be used in fitting ["flexible", "rigid", "single", "xyparaboloid",
            "thetaparaboloid", "fixedtheta"]
            angle: Azimuthal span of the panel
            iring: Which ring is the panel in
            ipanel: Panel number clockwise from top
            inrad: Radius at panel inner side
            ourad: Radius at panel outer side
        """
        self.kind = kind
        self.ipanel = ipanel + 1
        self.iring = iring + 1
        self.inrad = inrad
        self.ourad = ourad
        self.theta1 = ipanel * angle
        self.theta2 = (ipanel + 1) * angle
        self.zeta = (ipanel + 0.5) * angle
        self.solved = False
        self.bmp = [inrad * np.sin(self.zeta), -inrad * np.cos(self.zeta)]
        self.tmp = [ourad * np.sin(self.zeta), -ourad * np.cos(self.zeta)]
        rt = (self.inrad + self.ourad) / 2
        self.center = [rt * np.sin(self.zeta), rt * np.cos(self.zeta)]
        self.screws = np.ndarray([4, 2])

        rscale = 0.1 * (ourad - inrad)
        tscale = 0.1 * angle
        self.screws[0, :] = np.sin(self.theta1 + tscale), np.cos(self.theta1 + tscale)
        self.screws[1, :] = np.sin(self.theta2 - tscale), np.cos(self.theta2 - tscale)
        self.screws[2, :] = np.sin(self.theta1 + tscale), np.cos(self.theta1 + tscale)
        self.screws[3, :] = np.sin(self.theta2 - tscale), np.cos(self.theta2 - tscale)
        self.screws[0, :] *= (inrad + rscale)
        self.screws[1, :] *= (inrad + rscale)
        self.screws[2, :] *= (ourad - rscale)
        self.screws[3, :] *= (ourad - rscale)

        self.nsamp = 0
        self.values = []
        self.corr = None

        if self.kind == "rigid":
            self.solve = self._solve_rigid
            self.corr_point = self._corr_point_rigid
        elif self.kind == "single":
            self.solve = self._solve_single
            self.corr_point = self._corr_point_single
        elif self.kind == "xyparaboloid":
            self.solve = self._solve_scipy
            self.corr_point = self._corr_point_flexi_scipy
            self._paraboloid = self._xyaxes_paraboloid
        elif self.kind == "thetaparaboloid":
            self.solve = self._solve_scipy
            self.corr_point = self._corr_point_flexi_scipy
            self._paraboloid = self._rotated_paraboloid
        elif self.kind == "fixedtheta":
            self.solve = self._solve_scipy
            self.corr_point = self._corr_point_flexi_scipy
            self._paraboloid = self._fixed_paraboloid
        else:
            raise Exception("Unknown panel kind: ", self.kind)

    def is_inside(self, rad, phi):
        """
        Check if a point is inside a panel using polar coordinates
        Args:
            rad: radius of the point
            phi: angle of the point in polar coordinates

        Returns:
        True if point is inside the panel, False otherwise
        """
        # Simple test of polar coordinates to check that a point is
        # inside this panel
        angle = self.theta1 <= phi <= self.theta2
        radius = self.inrad <= rad <= self.ourad
        return angle and radius

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

    def _fixed_paraboloid(self, coords, ucurv, vcurv, zoff):
        """
        Surface model to be used in fitting with scipy
        Same as the rotated paraboloid below, but theta is the panel center angle
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

    def _solve_rigid(self):
        """
        Fit panel surface using AIPS gauss elimination model for rigid panels
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

    def _solve_single(self):
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

    def _corr_point_flexi_scipy(self, xcoor, ycoor):
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
        Computes fitted value for point [xcoor, ycoor] using AIPS gauss elimination model for rigid panels
        Args:
            xcoor: X coordinate of point
            ycoor: Y coordinate of point

        Returns:
        Fitted value at xcoor,ycoor
        """
        return xcoor * self.par[0] + ycoor * self.par[1] + self.par[2]

    def _corr_point_single(self, xcoor, ycoor):
        """
        Computes fitted value for point [xcoor, ycoor] using AIPS shift only panels
        Args:
            xcoor: X coordinate of point
            ycoor: Y coordinate of point

        Returns:
        Fitted value at xcoor,ycoor
        """
        return self.par[0]

    def export_adjustments(self, unit='mm', screen=False):
        """
        Exports panel screw adjustments to a string
        Args:
            unit: Unit for screw adjustments ['mm','miliinches']
            screen: display values in terminal

        Returns:
        String with screw adjustments for this panel
        """
        if unit == 'mm':
            fac = 1.0
        elif unit == 'miliinches':
            fac = 1000.0 / 25.4
        else:
            raise Exception("Unknown unit: " + unit)

        string = '{0:8d} {1:8d}'.format(self.iring, self.ipanel)
        for screw in self.screws[:, ]:
            string += ' {0:10.2f}'.format(fac * self.corr_point(*screw))
        if screen:
            print(string)
        return string

    def print_misc(self, verbose=False):
        """
        Print miscelaneous information about the panel to the terminal
        Args:
            verbose: Include more information in print
        """
        print("########################################")
        print("{0:20s}={1:8d}".format("ipanel", self.ipanel))
        print("{0:20s}={1:8s}".format("kind", " " + self.kind))
        print("{0:20s}={1:8.5f}".format("inrad", self.inrad))
        print("{0:20s}={1:8.5f}".format("ourad", self.ourad))
        print("{0:20s}={1:8.5f}".format("theta1", self.theta1))
        print("{0:20s}={1:8.5f}".format("theta2", self.theta2))
        print("{0:20s}={1:8.5f}".format("zeta", self.zeta))
        print("{0:20s}={1:8.5f}, {2:8.5f}".format("bmp", *self.bmp))
        print("{0:20s}={1:8.5f}, {2:8.5f}".format("tmp", *self.tmp))
        print("{0:20s}={1:8d}".format("nsamp", self.nsamp))
        if verbose:
            for isamp in range(self.nsamp):
                strg = "{0:20s}=".format("samp{0:d}".format(isamp))
                for val in self.values[isamp]:
                    strg += str(val) + ", "
                print(strg)
        print()

    def plot(self, ax, screws=False):
        """
        Plot panel outline to ax
        Args:
            ax: matplotlib axes instance
            screws: Display screws in plot

        """
        lw = 0.5
        msize = 2
        x1 = self.inrad * np.sin(self.theta1)
        y1 = self.inrad * np.cos(self.theta1)
        x2 = self.ourad * np.sin(self.theta1)
        y2 = self.ourad * np.cos(self.theta1)
        ax.plot([x1, x2], [y1, y2], ls='-', color='black', marker=None, lw=lw)
        rt = (self.inrad + self.ourad) / 2
        xt = rt * np.sin(self.zeta)
        yt = rt * np.cos(self.zeta)
        ax.text(xt, yt, str(self.ipanel), fontsize=5, ha='center')
        if screws:
            markers = ['x', 'o', '*', '+']
            colors = ['g', 'g', 'r', 'r']
            for iscrew in range(self.screws.shape[0]):
                screw = self.screws[iscrew, ]
                ax.scatter(screw[0], screw[1], marker=markers[iscrew],
                           lw=lw, s=msize, color=colors[iscrew])
        if self.ipanel == 1:
            # Draw ring outline with first panel
            inrad = plt.Circle((0, 0), self.inrad, color='black', fill=False, lw=lw)
            ourad = plt.Circle((0, 0), self.ourad, color='black', fill=False, lw=lw)
            ax.add_patch(inrad)
            ax.add_patch(ourad)
