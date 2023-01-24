import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import optimize as opt

lnbr = "\n"


# static methods not linked to any specific class
def gauss_numpy(system, vector):
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


def convert_to_db(val: float):
    """
    Converts a float value to decibels
    Args:
        val (float): Value to be converted to decibels
    Returns:
        Value in decibels
    """
    return 10.0 * np.log10(val)


def read_fits(filename):
    """
    Reads a FITS file and do sanity checks on its dimensionality
    Args:
        filename: a string containing the FITS file name/path

    Returns:
    The FITS header and the associated data array
    """
    hdul = fits.open(filename)
    head = hdul[0].header
    if head["NAXIS"] != 2:
        if head["NAXIS"] < 2:
            raise Exception(filename + " is not bi-dimensional")
        elif head["NAXIS"] > 2:
            for iax in range(2, head["NAXIS"]):
                if head["NAXIS" + str(iax + 1)] != 1:
                    raise Exception(filename + " is not bi-dimensional")
    if head["NAXIS1"] != head["NAXIS2"]:
        raise Exception(filename + " image is not square")

    data = hdul[0].data[0, 0, :, :]
    hdul.close()
    return head, data


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

        # AIPS definition of the screws, seem arbitrary and don't
        # really work
        # self.screws[0,:] = -self.bmp[0],self.bmp[1]
        # self.screws[1,:] =  self.bmp[0],self.bmp[1]
        # self.screws[2,:] = -self.tmp[0],self.tmp[1]
        # self.screws[3,:] =  self.tmp[0],self.tmp[1]
        rscale = 0.1 * (ourad - inrad)
        tscale = 0.1 * angle
        self.screws[0, :] = np.sin(self.theta1 + tscale), np.cos(self.theta1 + tscale)
        self.screws[1, :] = np.sin(self.theta2 - tscale), np.cos(self.theta2 - tscale)
        self.screws[2, :] = np.sin(self.theta1 + tscale), np.cos(self.theta1 + tscale)
        self.screws[3, :] = np.sin(self.theta2 - tscale), np.cos(self.theta2 - tscale)
        self.screws[0, :] *= inrad + rscale
        self.screws[1, :] *= inrad + rscale
        self.screws[2, :] *= ourad - rscale
        self.screws[3, :] *= ourad - rscale

        self.nsamp = 0
        self.values = []
        self.corr = None

        if self.kind == "flexible":
            self.solve = self._solve_flexi
            self.corr_point = self._corr_point_flexi
        elif self.kind == "rigid":
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

    def _solve_flexi(self):
        """
        Fit panel surface using AIPS gauss elimination model for flexible panels
        """
        syssize = 4
        if self.nsamp < syssize:
            # In this case the matrix will always be singular as the
            # rows will be linear combinations
            return
        system = np.zeros([syssize, syssize])
        vector = np.zeros(syssize)

        for ipoint in range(len(self.values)):
            dev = self.values[ipoint][-1]
            if dev != 0:
                xcoor = self.values[ipoint][0]
                ycoor = self.values[ipoint][1]
                fac = self.bmp[0] + ycoor * (self.tmp[0] - self.bmp[0]) / self.tmp[1]
                coef1 = (
                    (self.tmp[1] - ycoor) * (1.0 - xcoor / fac) / (2.0 * self.tmp[1])
                )
                coef2 = ycoor * (1.0 - xcoor / fac) / (2.0 * self.tmp[1])
                coef3 = (
                    (self.tmp[1] - ycoor) * (1.0 + xcoor / fac) / (2.0 * self.tmp[1])
                )
                coef4 = ycoor * (1.0 + xcoor / fac) / (2.0 * self.tmp[1])
                system[0, 0] += coef1 * coef1
                system[0, 1] += coef1 * coef2
                system[0, 2] += coef1 * coef3
                system[0, 3] += coef1 * coef4
                system[1, 0] = system[0, 1]
                system[1, 1] += coef2 * coef2
                system[1, 2] += coef2 * coef3
                system[1, 3] += coef2 * coef4
                system[2, 0] = system[0, 2]
                system[2, 1] = system[1, 2]
                system[2, 2] += coef3 * coef3
                system[2, 3] += coef3 * coef4
                system[3, 0] = system[0, 3]
                system[3, 1] = system[1, 3]
                system[3, 2] = system[2, 3]
                system[3, 3] += coef4 * coef4
                vector[0] = vector[0] + dev * coef1
                vector[1] = vector[1] + dev * coef2
                vector[2] = vector[2] + dev * coef3
                vector[3] = vector[3] + dev * coef4

        self.par = gauss_numpy(system, vector)
        self.solved = True

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
                result = opt.curve_fit(
                    self._paraboloid,
                    coords,
                    devia,
                    p0=p0,
                    bounds=[liminf, limsup],
                    maxfev=maxfev,
                )
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
        return (
            -(((x - self.center[0]) / xcurv) ** 2 + ((y - self.center[1]) / ycurv) ** 2)
            + zoff
        )

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

        self.par = gauss_numpy(system, vector)
        self.solved = True
        return

    def _solve_single(self):
        """
        Fit panel surface as a simple mean of its points Z deviation
        """
        if self.nsamp > 0:
            # Solve panel adjustments for rigid vertical shift only panels
            self.par = np.zeros(1)
            shiftmean = 0.0
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

    def _corr_point_flexi(self, xcoor, ycoor):
        """
        Computes fitted value for point [xcoor, ycoor] using AIPS gauss elimination model for flexible panels
        Args:
            xcoor: X coordinate of point
            ycoor: Y coordinate of point

        Returns:
        Fitted value at xcoor,ycoor
        """
        coef = np.ndarray([4])
        corrval = 0
        fac = self.bmp[0] + ycoor * (self.tmp[0] - self.bmp[0]) / self.tmp[1]
        coef[0] = (self.tmp[1] - ycoor) * (1.0 - xcoor / fac) / (2.0 * self.tmp[1])
        coef[1] = ycoor * (1.0 - xcoor / fac) / (2.0 * self.tmp[1])
        coef[2] = (self.tmp[1] - ycoor) * (1.0 + xcoor / fac) / (2.0 * self.tmp[1])
        coef[3] = ycoor * (1.0 + xcoor / fac) / (2.0 * self.tmp[1])
        for ipar in range(len(self.par)):
            corrval += coef[ipar] * self.par[ipar]
        return corrval

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

    def export_adjustments(self, unit="mm", screen=False):
        """
        Exports panel screw adjustments to a string
        Args:
            unit: Unit for screw adjustments ['mm','miliinches']
            screen: display values in terminal

        Returns:
        String with screw adjustments for this panel
        """
        if unit == "mm":
            fac = 1.0
        elif unit == "miliinches":
            fac = 1000.0 / 25.4
        else:
            raise Exception("Unknown unit: " + unit)

        string = "{0:8d} {1:8d}".format(self.iring, self.ipanel)
        for screw in self.screws[
            :,
        ]:
            string += " {0:10.2f}".format(fac * self.corr_point(*screw))
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
        ax.plot([x1, x2], [y1, y2], ls="-", color="black", marker=None, lw=lw)
        rt = (self.inrad + self.ourad) / 2
        xt = rt * np.sin(self.zeta)
        yt = rt * np.cos(self.zeta)
        ax.text(xt, yt, str(self.ipanel), fontsize=5, ha="center")
        if screws:
            markers = ["x", "o", "*", "+"]
            colors = ["g", "g", "r", "r"]
            for iscrew in range(self.screws.shape[0]):
                screw = self.screws[
                    iscrew,
                ]
                ax.scatter(
                    screw[0],
                    screw[1],
                    marker=markers[iscrew],
                    lw=lw,
                    s=msize,
                    color=colors[iscrew],
                )
        if self.ipanel == 1:
            # Draw ring outline with first panel
            inrad = plt.Circle((0, 0), self.inrad, color="black", fill=False, lw=lw)
            ourad = plt.Circle((0, 0), self.ourad, color="black", fill=False, lw=lw)
            ax.add_patch(inrad)
            ax.add_patch(ourad)


class AntennaSurface:
    def __init__(self, amp, dev, telescope, cutoff=0.21, pkind=None):
        """
        Antenna Surface description capable of computing RMS, Gains, and fitting the surface to obtain screw adjustments
        Args:
            amp: Amplitude aperture image (AIPS FITS file)
            dev: Physical deviation aperture image (AIPS FITS file)
            telescope: currently supported: ["VLA","VLBA"]
            cutoff: fractional cutoff on the amplitude image to exclude regions with weak amplitude from the panel
            surface fitting
            pkind: Kind of panel surface fitting, if is None defaults to telescope default
        """
        # Initializes antenna surface parameters
        self.ampfile = amp
        self.devfile = dev

        self.ingains = np.nan
        self.ougains = np.nan
        self.inrms = np.nan
        self.ourms = np.nan

        self._read_images()
        self.cut = cutoff * np.max(self.amp)
        print(self.cut)

        if telescope == "VLA":
            self._init_vla()
        elif telescope == "VLBA":
            self._init_vlba()
        else:
            raise Exception("Unknown telescope: " + telescope)
        if pkind is None:
            pass
        else:
            self.panelkind = pkind

        self._get_aips_headpars()
        self.reso = self.diam / self.npoint

        self.resi = None
        self.corr = None
        self.solved = False
        if self.ringed:
            self._build_polar()
            self._build_ring_panels()
            self._build_ring_mask()
            self.fetch_panel = self._fetch_panel_ringed
            self.compile_panel_points = self._compile_panel_points_ringed

    def _get_aips_headpars(self):
        """
        Fetches AIPS specific metadata from FITS headers
        """
        for line in self.devhead["HISTORY"]:
            wrds = line.split()
            if wrds[1] == "Visibilities":
                self.npoint = np.sqrt(int(wrds[-1]))
            elif wrds[1] == "Observing":
                # Stored in mm
                self.wavel = 1000 * float(wrds[-2])
            elif wrds[1] == "Antenna" and wrds[2] == "surface":
                self.inlim = abs(float(wrds[-3]))
                self.oulim = abs(float(wrds[-2]))

    def _read_images(self):
        """
        Reads amplitude and deviation images and initializes the X and Y axes
        """
        self.amphead, self.amp = read_fits(self.ampfile)
        self.devhead, self.dev = read_fits(self.devfile)
        self.dev *= 1000
        #
        if self.devhead["NAXIS1"] != self.amphead["NAXIS1"]:
            raise Exception("Amplitude and deviation images have different sizes")
        self.npix = int(self.devhead["NAXIS1"])
        self.xaxis = LinearAxis(
            self.npix,
            self.amphead["CRPIX1"],
            self.amphead["CRVAL1"],
            self.amphead["CDELT1"],
        )
        self.yaxis = LinearAxis(
            self.npix,
            self.amphead["CRPIX2"],
            self.amphead["CRVAL2"],
            self.amphead["CDELT2"],
        )
        return

    def _build_ring_mask(self):
        """
        Builds the mask on regions to be included in panel surface masks, specific to circular antennas as there is an
        outer and inner limit to the mask based on the antenna's inner receiver hole and outer edge
        """
        self.mask = np.where(self.amp < self.cut, False, True)
        self.mask = np.where(self.rad > self.inlim, self.mask, False)
        self.mask = np.where(self.rad < self.oulim, self.mask, False)
        self.mask = np.where(np.isnan(self.dev), False, self.mask)

    # Other known telescopes should be included here, ALMA, ngVLA
    def _init_vla(self):
        """
        Initializes object according to parameters specific to VLA panel distribution
        """
        self.panelkind = "flexible"
        self.telescope = "VLA"
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
        self.panelkind = "flexible"
        self.telescope = "VLBA"
        self.diam = 25.0  # meters
        self.focus = 8.75  # meters
        self.ringed = True
        self.nrings = 6
        self.npanel = [20, 20, 40, 40, 40, 40]
        self.inrad = [1.676, 3.518, 5.423, 7.277, 9.081, 10.808]
        self.ourad = [3.518, 5.423, 7.277, 9.081, 10.808, 12.500]
        self.inlim = 2.0
        self.oulim = 12.0

    def _build_polar(self):
        """
        Build polar coordinate grid, specific for circular antennas with panels arranged in rings
        """
        self.rad = np.zeros([self.npix, self.npix])
        self.phi = np.zeros([self.npix, self.npix])
        for iy in range(self.npix):
            ycoor = self.yaxis.idx_to_coor(iy + 0.5)
            for ix in range(self.npix):
                xcoor = self.xaxis.idx_to_coor(ix + 0.5)
                self.rad[ix, iy] = np.sqrt(xcoor**2 + ycoor**2)
                self.phi[ix, iy] = np.arctan2(ycoor, xcoor)
                if self.phi[ix, iy] < 0:
                    self.phi[ix, iy] += 2 * np.pi

    def _build_ring_panels(self):
        """
        Build list of panels, specific for circular antennas with panels arranged in rings
        """
        self.panels = []
        for iring in range(self.nrings):
            angle = 2.0 * np.pi / self.npanel[iring]
            for ipanel in range(self.npanel[iring]):
                panel = RingPanel(
                    self.panelkind,
                    angle,
                    iring,
                    ipanel,
                    self.inrad[iring],
                    self.ourad[iring],
                )
                self.panels.append(panel)
        return

    def _compile_panel_points_ringed(self):
        """
        Loops through the points in the antenna surface and checks to which panels it belongs,
        specific for circular antennas with panels arranged in rings
        """
        for iy in range(self.npix):
            yc = self.yaxis.idx_to_coor(iy + 0.5)
            for ix in range(self.npix):
                if self.mask[ix, iy]:
                    xc = self.xaxis.idx_to_coor(ix + 0.5)
                    # How to do the coordinate choice here without
                    # adding an if?
                    for panel in self.panels:
                        if panel.is_inside(self.rad[ix, iy], self.phi[ix, iy]):
                            panel.add_point([xc, yc, ix, iy, self.dev[ix, iy]])

    def _fetch_panel_ringed(self, ring, panel):
        """
        Fetch a panel object from the panel list using its ring and panel numbers,
        specific for circular antennas with panels arranged in rings
        Args:
            ring: Ring number
            panel: Panel number

        Returns:
        Panel object
        """
        if ring == 1:
            ipanel = panel - 1
        else:
            ipanel = np.sum(self.npanel[: ring - 1]) + panel - 1
        return self.panels[ipanel]

    def gains(self):
        """
        Computes antenna gains in decibels before and after panel surface fitting
        Returns:
        Gains before panel fitting OR Gains before and after panel fitting
        """
        self.ingains = self._gains_array(self.dev)
        if self.resi is None:
            return self.ingains
        else:
            self.ougains = self._gains_array(self.resi)
            return self.ingains, self.ougains

    def _gains_array(self, arr):
        """
        Worker for gains method, works with the actual arrays to compute the gains
        Args:
            arr: Deviation image over which to compute the gains

        Returns:
        Actual and theoretical gains
        """
        # Compute the actual and theoretical gains for the current
        # antenna surface. What is the unit for the wavelength? mm
        forpi = 4.0 * np.pi
        fact = 1000.0 * self.reso / self.wavel
        fact *= fact
        #
        # What are these sums?
        sumrad = 0.0
        sumtheta = 0.0
        nsamp = 0
        #    convert surface error to phase
        #    and compute gain loss
        for iy in range(self.npix):
            for ix in range(self.npix):
                if self.mask[ix, iy]:
                    quo = self.rad[ix, iy] / (2.0 * self.focus)
                    phase = (
                        arr[ix, iy] * forpi / (np.sqrt(1.0 + quo * quo) * self.wavel)
                    )
                    sumrad += np.cos(phase)
                    sumtheta += np.sin(phase)
                    nsamp += 1

        ampmax = np.sqrt(sumrad * sumrad + sumtheta * sumtheta)
        if nsamp <= 0:
            raise Exception("Antenna is blanked")
        ampmax *= fact / nsamp
        gain = ampmax * forpi
        thgain = fact * forpi
        #
        gain = convert_to_db(gain)
        thgain = convert_to_db(thgain)
        return gain, thgain

    def get_rms(self):
        """
        Computes antenna surface RMS before and after panel surface fitting
        Returns:
        RMS before panel fitting OR RMS before and after panel fitting
        """
        self.inrms = np.sqrt(np.mean(self.dev[self.mask] ** 2))
        if self.resi is None:
            return self.inrms
        else:
            self.ourms = np.sqrt(np.mean(self.resi[self.mask] ** 2))
            return self.inrms, self.ourms

    def fit_surface(self):
        """
        Loops over the panels to fit the panel surfaces
        """
        for panel in self.panels:
            panel.solve()
        self.solved = True

    def correct_surface(self):
        """
        Apply corrections determined by the panel surface fitting methods to the antenna surface
        """
        if not self.solved:
            raise Exception("Panels must be fitted before atempting a correction")
        self.corr = np.where(self.mask, 0, np.nan)
        self.resi = np.copy(self.dev)
        for panel in self.panels:
            panel.get_corrections()
            for ipnt in range(len(panel.corr)):
                val = panel.values[ipnt]
                ix, iy = int(val[2]), int(val[3])
                self.resi[ix, iy] -= panel.corr[ipnt]
                self.corr[ix, iy] = -panel.corr[ipnt]

    def print_misc(self):
        """
        Print miscelaneous information on the panels in the antenna surface
        """
        for panel in self.panels:
            panel.print_misc()

    def plot_surface(self, filename=None, mask=False, screws=False):
        """
        Do plots of the antenna surface
        Args:
            filename: Save plot to a file rather than displaying it with matplotlib widgets
            mask: Display mask and amplitudes rather than deviation images
            screws: Display the screws on the panels
        """
        vmin, vmax = np.nanmin(self.dev), np.nanmax(self.dev)
        rms = self.get_rms()
        if mask:
            fig, ax = plt.subplots(1, 2, figsize=[10, 5])
            title = "Mask"
            self._plot_surface(
                self.mask, title, fig, ax[0], 0, 1, screws=screws, mask=mask
            )
            vmin, vmax = np.nanmin(self.amp), np.nanmax(self.amp)
            title = "Amplitude min={0:.5f}, max ={1:.5f}".format(vmin, vmax)
            self._plot_surface(
                self.amp,
                title,
                fig,
                ax[1],
                vmin,
                vmax,
                screws=screws,
                unit=self.amphead["BUNIT"].strip(),
            )
        else:
            if self.resi is None:
                fig, ax = plt.subplots()
                title = "Before correction\nRMS = {0:8.5} mm".format(rms)
                self._plot_surface(self.dev, title, fig, ax, vmin, vmax, screws=screws)
            else:
                fig, ax = plt.subplots(1, 3, figsize=[15, 5])
                title = "Before correction\nRMS = {0:.3} mm".format(rms[0])
                self._plot_surface(
                    self.dev, title, fig, ax[0], vmin, vmax, screws=screws
                )
                title = "Corrections"
                self._plot_surface(
                    self.corr, title, fig, ax[1], vmin, vmax, screws=screws
                )
                title = "After correction\nRMS = {0:.3} mm".format(rms[1])
                self._plot_surface(
                    self.resi, title, fig, ax[2], vmin, vmax, screws=screws
                )
        fig.suptitle("Antenna Surface")
        fig.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, dpi=600)

    def _plot_surface(
        self, data, title, fig, ax, vmin, vmax, screws=False, mask=False, unit="mm"
    ):
        """
        Does the plotting of a data array in a figure's subplot
        Args:
            data: The array to be plotted
            title: Title of the subplot
            fig: Global figure containing the subplots
            ax: matplotlib axes instance describing the subplot
            vmin: minimum to the color scale
            vmax: maximum to the color scale
            screws: Display screws
            mask: do not add colorbar if plotting a mask
            unit: Unit of the data in the color scale
        """
        ax.set_title(title)
        # set the limits of the plot to the limits of the data
        xmin = self.xaxis.idx_to_coor(-0.5)
        xmax = self.xaxis.idx_to_coor(self.xaxis.n - 0.5)
        ymin = self.yaxis.idx_to_coor(-0.5)
        ymax = self.yaxis.idx_to_coor(self.yaxis.n - 0.5)
        im = ax.imshow(
            np.flipud(data),
            cmap="viridis",
            interpolation="nearest",
            extent=[xmin, xmax, ymin, ymax],
            vmin=vmin,
            vmax=vmax,
        )
        divider = make_axes_locatable(ax)
        if not mask:
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, label="Z Scale [" + unit + "]", cax=cax)
        ax.set_xlabel("X axis [m]")
        ax.set_ylabel("Y axis [m]")
        for panel in self.panels:
            panel.plot(ax, screws=screws)

    def export_corrected(self, filename):
        """
        Export corrected surface to a FITS file
        Args:
            filename: Output FITS file name/path
        """
        if self.resi is None:
            raise Exception("Cannot export corrected surface")
        hdu = fits.PrimaryHDU(self.resi)
        hdu.header = self.devhead
        hdu.header["ORIGIN"] = "Astrohack PANEL"
        hdu.writeto(filename, overwrite=True)
        return

    def export_screw_adjustments(self, filename, unit="mm"):
        """
        Export screw adjustments for all panels onto an ASCII file
        Args:
            filename: ASCII file name/path
            unit: unit for panel screw adjustments ['mm','miliinches']
        """
        spc = " "
        outfile = "Screw adjustments for {0:s} {1:s} antenna\n".format(
            self.telescope, self.amphead["telescop"]
        )
        outfile += "Adjustments are in " + unit + lnbr
        outfile += 2 * lnbr
        outfile += 25 * spc + "{0:22s}{1:22s}".format("Inner Edge", "Outer Edge") + lnbr
        outfile += 5 * spc + "{0:8s}{1:8s}".format("Ring", "panel")
        outfile += 2 * spc + 2 * "{0:11s}{1:11s}".format("left", "right") + lnbr
        for panel in self.panels:
            outfile += panel.export_adjustments(unit=unit) + lnbr
        lefile = open(filename, "w")
        lefile.write(outfile)
        lefile.close()
