from scipy import optimize as opt

from astrohack._utils._algorithms import _gauss_elimination_numpy, _least_squares_fit
from astrohack._utils._constants import *
from astrohack._utils._conversion import _convert_unit
from astrohack._utils._logger._astrohack_logger import _get_astrohack_logger

panel_models = ["mean", "rigid", "corotated_scipy", "corotated_lst_sq", "corotated_robust", "xy_paraboloid",
                "rotated_paraboloid", "full_paraboloid_lst_sq"]
imean = 0
irigid = 1
icorscp = 2
icorlst = 3
icorrob = 4
ixypara = 5
irotpara = 6
ifulllst = 7

warned = False


def set_warned(value: bool):
    """
    set the global warned to avoid repeated messages about experimental values
    Args:
        value: True or False

    """
    global warned
    warned = value


class BasePanel:
    markers = ['X', 'o', '*', 'P', 'D']
    colors = ['g', 'g', 'r', 'r', 'b']
    fontsize = 4
    linewidth = 0.5
    markersize = 2
    linecolor = 'black'

    def __init__(self, model, screws, label, center=None, zeta=None):
        """
        Initializes the base panel with the appropriated fitting methods and providing basic functionality
        Fitting method models are:
        AIPS fitting models:
            mean: The panel is corrected by the mean of its samples
            rigid: The panel samples are fitted to a rigid surface
        Corotated Paraboloids (the two bending axes are parallel and perpendicular to the radius of the antenna crossing
        the middle of the panel):
            corotated_scipy: Paraboloid is fitted using scipy.optimize, robust but slow
            corotated_lst_sq: Paraboloid is fitted using the linear algebra least squares method, fast but unreliable
            corotated_robust: Tries corotated_lst_sq, if it diverges falls back to corotated_scipy
        Experimental fitting models:
            xy_paraboloid: fitted using scipy.optimize, bending axes are parallel to the x and y axes
            rotated_paraboloid: fitted using scipy.optimize, bending axes can be rotated by an arbitrary angle
            full_paraboloid_lst_sq: Full 9 parameter paraboloid fitted using least_squares method, heavily overfits
        Args:
            model: What model of surface fitting method to be used
            label: Panel label
            screws: position of the screws
            center: Panel center
            zeta: panel center angle
        """
        self.model = model
        self.solved = False
        self.label = label
        self.screws = screws
        self.samples = []
        self.margins = []
        self.corr = None

        if center is None:
            self.center = [0, 0]
        else:
            self.center = center
        if zeta is None:
            self.zeta = 0
        else:
            self.zeta = zeta
        self._associate()

    def _associate(self):
        """
        Does the fitting method associations according to the model chosen by the user
        """
        logger = _get_astrohack_logger()
        try:
            imodel = panel_models.index(self.model)
        except ValueError:
            logger.error("Unknown panel model: "+self.model)
            raise ValueError('Panel model not in list')
        if imodel > icorrob:
            self._warn_experimental_method()
        if imodel == irigid:
            self._associate_rigid()
        elif imodel == imean:
            self._associate_mean()
        elif imodel == ixypara:
            self._associate_scipy(self._xyaxes_paraboloid, 3)
        elif imodel == irotpara:
            self._associate_scipy(self._rotated_paraboloid, 4)
        elif imodel == icorscp:
            self._associate_scipy(self._corotated_paraboloid, 3)
        elif imodel == ifulllst:
            self._associate_least_squares()
        elif imodel == icorlst:
            self._associate_corotated_lst_sq()
        elif imodel == icorrob:
            self._associate_robust()
        
    def _warn_experimental_method(self):
        """
        Raises a warning about experimental methods if a warning has not been raised before
        """
        if warned:
            return
        else:
            logger = _get_astrohack_logger()
            logger.warning("Experimental model: "+self.model)
            set_warned(True)

    def _associate_scipy(self, fitting_function, NPAR):
        """
        Associate the proper methods to enable scipy fitting
        Args:
            fitting_function: The fitting function to be used by scipy
            NPAR: Number of paramenters in the fitting function
        """
        self.NPAR = NPAR
        self._solve_sub = self._solve_scipy
        self.corr_point = self._corr_point_scipy
        self._fitting_function = fitting_function

    def _associate_robust(self):
        """
        Associate fitting method for the corotated_robust model
        Returns:

        """
        self.NPAR = 3
        self._solve_sub = self._solve_robust
        self.corr_point = self._corr_point_corotated_lst_sq
        self._fitting_function = self._corotated_paraboloid

    def _associate_rigid(self):
        """
        Associate the proper methods to enable the rigid panel Linear algebra fitting
        """
        self.NPAR = 3
        self._solve_sub = self._solve_rigid
        self.corr_point = self._corr_point_rigid

    def _associate_mean(self):
        """
        Associate the proper methods to enable fitting by mean determination
        """
        self.NPAR = 1
        self._solve_sub = self._solve_mean
        self.corr_point = self._corr_point_mean

    def _associate_least_squares(self):
        """
        Associate the proper methods to enable least squares fitting of a fully fledged 9 parameter paraboloid
        """
        self.NPAR = 9
        self._solve_sub = self._solve_least_squares_paraboloid
        self.corr_point = self._corr_point_least_squares_paraboloid

    def _associate_corotated_lst_sq(self):
        """
        Associate the proper methods to enable least squares fitting of a corotated paraboloid
        """
        self.NPAR = 3
        self._solve_sub = self._solve_corotated_lst_sq
        self.corr_point = self._corr_point_corotated_lst_sq

    def add_sample(self, value):
        """
        Add a point to the panel's list of points to be fitted
        Args:
            value: tuple/list containing point description [xcoor,ycoor,xidx,yidx,value]
        """
        self.samples.append(value)

    def add_margin(self, value):
        """
        Add a point to the panel's list of points to be corrected, but not fitted
        Args:
            value: tuple/list containing point description [xcoor,ycoor,xidx,yidx,value]
        """
        self.margins.append(value)

    def solve(self):
        """
        Wrapping method around fitting to allow for a fallback to mean fitting in the case of an impossible fit

        Returns:
            True: in case of successful fit
            False: in case of fallback fit
        """
        # fallback behaviour for impossible fits
        if len(self.samples) < self.NPAR:
            self._fallback_solve()
            status = False
        else:
            try:
                self._solve_sub()
                status = True
            except np.linalg.LinAlgError:
                self._fallback_solve()
                status = False
        return status

    def _fallback_solve(self):
        """
        Changes the method association to mean surface fitting, and fits the panel with it
        """
        self._associate_mean()
        self._solve_sub()

    def _solve_least_squares_paraboloid(self):
        """
        Builds the designer matrix for least squares fitting, and calls the _least_squares fitter for a fully fledged
        9 parameter paraboloid
        """
        # ax2y2 + bx2y + cxy2 + dx2 + ey2 + gxy + hx + iy + j
        data = np.array(self.samples)
        system = np.full((len(self.samples), self.NPAR), 1.0)
        system[:, 0] = data[:, 0]**2 * data[:, 1]**2
        system[:, 1] = data[:, 0]**2 * data[:, 1]
        system[:, 2] = data[:, 1]**2 * data[:, 0]
        system[:, 3] = data[:, 0] ** 2
        system[:, 4] = data[:, 1] ** 2
        system[:, 5] = data[:, 0] * data[:, 1]
        system[:, 6] = data[:, 0]
        system[:, 7] = data[:, 1]
        vector = data[:, -1]
        self.par, _, _ = _least_squares_fit(system, vector)
        self.solved = True

    def _corr_point_least_squares_paraboloid(self, xcoor, ycoor):
        """
        Computes the correction from the fitted parameters to the 9 parameter paraboloid at (xcoor, ycoor)
        Args:
            xcoor: Coordinate of point in X
            ycoor: Coordinate of point in Y
        Returns:
            The correction at point
        """
        # ax2y2 + bx2y + cxy2 + dx2 + ey2 + gxy + hx + iy + j
        xsq = xcoor**2
        ysq = ycoor**2
        point = self.par[0]*xsq*ysq + self.par[1]*xsq*ycoor + self.par[2]*ysq*xcoor
        point += self.par[3]*xsq + self.par[4]*ysq + self.par[5]*xcoor*ycoor
        point += self.par[6]*xcoor + self.par[7]*ycoor + self.par[8]
        return point

    def _solve_robust(self):
        """
        Try fitting the Surface of a panel using the corotated least_squares method, if that fails fallback to scipy
        fitting
        """
        try:
            self._solve_corotated_lst_sq()
        except np.linalg.LinAlgError:
            self._solve_scipy()

    def _solve_corotated_lst_sq(self):
        """
        Builds the designer matrix for least squares fitting, and calls the _least_squares fitter for a corotated
        paraboloid centered at the center of the panel
        """
        # a*u**2 + b*v**2 + c
        data = np.array(self.samples)
        system = np.full((len(self.samples), self.NPAR), 1.0)
        xc, yc = self.center
        system[:, 0] = ((data[:, 0] - xc) * np.cos(self.zeta) - (data[:, 1] - yc) * np.sin(self.zeta))**2  # U
        system[:, 1] = ((data[:, 0] - xc) * np.sin(self.zeta) + (data[:, 1] - yc) * np.cos(self.zeta))**2  # V
        vector = data[:, -1]
        self.par, _, _ = _least_squares_fit(system, vector)
        self.solved = True

    def _corr_point_corotated_lst_sq(self, xcoor, ycoor):
        """
        Computes the correction from the least squares fitted parameters to the corotated paraboloid
        Args:
            xcoor: Coordinate of point in X
            ycoor: Coordinate of point in Y
        Returns:
            The correction at point
        """
        # a*u**2 + b*v**2 + c
        xc, yc = self.center
        usq = ((xcoor - xc) * np.cos(self.zeta) - (ycoor - yc) * np.sin(self.zeta))**2
        vsq = ((xcoor - xc) * np.sin(self.zeta) + (ycoor - yc) * np.cos(self.zeta))**2
        return self.par[0]*usq + self.par[1]*vsq + self.par[2]

    def _solve_scipy(self, verbose=False, x0=None):
        """
        Fit ponel surface by using arbitrary models through scipy fitting engine
        Args:
            verbose: Increase verbosity in the fitting process
        """
        logger = _get_astrohack_logger()
        devia = np.ndarray([len(self.samples)])
        coords = np.ndarray([2, len(self.samples)])
        for i in range(len(self.samples)):
            devia[i] = self.samples[i][-1]
            coords[:, i] = self.samples[i][0], self.samples[i][1]

        liminf = [-np.inf, -np.inf, -np.inf]
        limsup = [np.inf, np.inf, np.inf]
        if x0 is None:
            p0 = [1e2, 1e2, np.mean(devia)]
        else:
            p0 = x0
        if self.model == panel_models[irotpara]:
            liminf.append(0.0)
            limsup.append(np.pi)
            p0.append(0)

        maxfevs = [100000, 1000000, 10000000]
        for maxfev in maxfevs:
            try:
                result = opt.curve_fit(self._fitting_function, coords, devia,
                                       p0=p0, bounds=[liminf, limsup],
                                       maxfev=maxfev)
            except RuntimeError:
                if verbose:
                    logger.info("Increasing number of iterations")
                continue
            else:
                self.par = result[0]
                self.solved = True
                if verbose:
                    logger.info("Converged with less than {0:d} iterations".format(maxfev))
                break

    def _xyaxes_paraboloid(self, coords, ucurv, vcurv, zoff):
        """
        Surface model to be used in fitting with scipy
        Assumes that the center of the paraboloid is the center of the panel
        In this model the panel can only bend in the x and y directions
        Args:
            coords: [x,y] coordinate pair for point
            ucurv: curvature in x direction
            vcurv: curvature in y direction
            zoff:  Z offset of the paraboloid

        Returns:
        Paraboloid value at X and Y
        """
        u = coords[0] - self.center[0]
        v = coords[1] - self.center[1]
        return ucurv * u**2 + vcurv * v**2 + zoff

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
        u = (x - xc) * np.cos(theta) - (y - yc) * np.sin(theta)
        v = (x - xc) * np.sin(theta) + (y - yc) * np.cos(theta)
        return ucurv * u**2 + vcurv * v**2 + zoff

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
        u = (x - xc) * np.cos(self.zeta) - (y - yc) * np.sin(self.zeta)
        v = (x - xc) * np.sin(self.zeta) + (y - yc) * np.cos(self.zeta)
        return ucurv * u**2 + vcurv * v**2 + zoff

    def _solve_rigid(self):
        """
        Fit panel surface using AIPS gaussian elimination model for rigid panels
        """
        system = np.zeros([self.NPAR, self.NPAR])
        vector = np.zeros(self.NPAR)
        for ipoint in range(len(self.samples)):
            if self.samples[ipoint][-1] != 0:
                system[0, 0] += self.samples[ipoint][0] * self.samples[ipoint][0]
                system[0, 1] += self.samples[ipoint][0] * self.samples[ipoint][1]
                system[0, 2] += self.samples[ipoint][0]
                system[1, 0] = system[0, 1]
                system[1, 1] += self.samples[ipoint][1] * self.samples[ipoint][1]
                system[1, 2] += self.samples[ipoint][1]
                system[2, 0] = system[0, 2]
                system[2, 1] = system[1, 2]
                system[2, 2] += 1.0
                vector[0] += self.samples[ipoint][-1] * self.samples[ipoint][0]
                vector[1] += self.samples[ipoint][-1] * self.samples[ipoint][1]
                vector[2] += self.samples[ipoint][-1]

        self.par = _gauss_elimination_numpy(system, vector)
        self.solved = True
        return

    def _solve_mean(self):
        """
        Fit panel surface as a simple mean of its points Z deviation
        """
        if len(self.samples) > 0:
            # Solve panel adjustments for rigid vertical shift only panels
            data = np.array(self.samples)[:, -1]
            self.par = [np.mean(data)]
        else:
            self.par = [0]
        self.solved = True
        return

    def get_corrections(self):
        """
        Store corrections for the points in the panel
        """
        if not self.solved:
            raise Exception("Cannot correct a panel that is not solved")
        lencorr = len(self.samples)+len(self.margins)
        self.corr = np.ndarray([lencorr, 3])
        icorr = 0
        for isamp in range(len(self.samples)):
            xc, yc = self.samples[isamp][0:2]
            ix, iy = self.samples[isamp][2:4]
            self.corr[icorr, :] = ix, iy, self.corr_point(xc, yc)
            icorr += 1
        for imarg in range(len(self.margins)):
            xc, yc = self.margins[imarg][0:2]
            ix, iy = self.margins[imarg][2:4]
            self.corr[icorr, :] = ix, iy, self.corr_point(xc, yc)
            icorr += 1
        return self.corr

    def _corr_point_scipy(self, xcoor, ycoor):
        """
        Computes the fitted value for point [xcoor, ycoor] using the scipy models
        Args:
            xcoor: X coordinate of point
            ycoor: Y coordinate of point

        Returns:
        Fitted value at xcoor,ycoor
        """
        corrval = self._fitting_function([xcoor, ycoor], *self.par)
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

    def export_screws(self, unit='mm'):
        """
        Export screw adjustments to a numpy array in unit
        Args:
            unit: Unit for the screw adjustments

        Returns:
            Numpy array with screw adjustments
        """
        fac = _convert_unit('m', unit, 'length')
        nscrew = len(self.screws)
        screw_corr = np.zeros(nscrew)
        for iscrew in range(nscrew):
            screw = self.screws[iscrew, :]
            screw_corr[iscrew] = fac*self.corr_point(*screw)
        return screw_corr

    def plot_label(self, ax, rotate=True):
        """
        Plots panel label to ax
        Args:
            ax: matplotlib axes instance
            rotate: Rotate label for better display
        """
        if rotate:
            angle = (-self.zeta % pi - pi/2)*_convert_unit('rad', 'deg', 'trigonometric')
        else:
            angle = 0
        ax.text(self.center[1], self.center[0], self.label, fontsize=self.fontsize, ha='center', va='center',
                rotation=angle)

    def plot_screws(self, ax):
        """
        Plots panel screws to ax
        Args:
            ax: matplotlib axes instance
        """
        for iscrew in range(self.screws.shape[0]):
            screw = self.screws[iscrew, ]
            ax.scatter(screw[1], screw[0], marker=self.markers[iscrew], lw=self.linewidth, s=self.markersize,
                       color=self.colors[iscrew])
