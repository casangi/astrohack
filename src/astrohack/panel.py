import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Arc
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import optimize as opt
from numba import jit

lnbr = '\n'

# static methods not linked to any specific class
def gauss_numpy(system,vector):
    inverse = np.linalg.inv(system)
    return np.dot(inverse,vector)


def convert_to_db(val):
    # Convert to decibels
    return 10.*np.log10(val)


def read_fits(filename):
    hdul = fits.open(filename)
    head = hdul[0].header
    if head['NAXIS'] != 2:
        if head['NAXIS'] < 2:
            raise Exception(filename+" is not bi-dimensional")
        elif head['NAXIS'] > 2:
            for iax in range(2,head['NAXIS']):
                if head['NAXIS'+str(iax+1)] != 1:
                    raise Exception(filename+" is not bi-dimensional")
    if head['NAXIS1'] != head['NAXIS2']:
        raise Exception(filename+" image is not square")

    if "AIPS" in hdul[0].header["ORIGIN"]:
        # AIPS data is in meters
        data = hdul[0].data[0,0,:,:]
    else:
        data = hdul[0].data[0,0,:,:]
    hdul.close()
    return head, data


class Linear_Axis:
    # According to JWS this class is superseeded by xarray, which
    # should be used instead
    def __init__(self,n,ref,val,inc):
        self.n = n
        self.ref = ref
        self.val = val
        self.inc = inc

    def idx_to_coor(self,idx):
        return (idx-self.ref)*self.inc + self.val

    def coor_to_idx(self,coor):
        return (coor-self.val)/self.inc +self.ref

    
class Ring_Panel:
    # This class describes and treats panels that are arranged in
    # rings on the Antenna surface
    
    def __init__(self,kind,angle,iring,ipanel,inrad,ourad):
        # Panel initialization
        self.kind   = kind
        self.ipanel = ipanel+1
        self.iring  = iring+1
        self.inrad  = inrad
        self.ourad  = ourad
        self.theta1 = ipanel*angle
        self.theta2 = (ipanel+1)*angle
        self.zeta   = (ipanel+0.5)*angle
        self.solved = False
        self.bmp = [inrad*np.sin(self.zeta),-inrad*np.cos(self.zeta)]
        self.tmp = [ourad*np.sin(self.zeta),-ourad*np.cos(self.zeta)]
        rt = (self.inrad+self.ourad)/2
        self.center = [rt*np.sin(self.zeta),rt*np.cos(self.zeta)]
        self.screws = np.ndarray([4,2])

        # AIPS definition of the screws, seem arbitrary and don't
        # really work
        # self.screws[0,:] = -self.bmp[0],self.bmp[1]
        # self.screws[1,:] =  self.bmp[0],self.bmp[1]
        # self.screws[2,:] = -self.tmp[0],self.tmp[1]
        # self.screws[3,:] =  self.tmp[0],self.tmp[1]
        rscale = 0.1*(ourad-inrad)
        tscale = 0.1*angle
        self.screws[0,:] = np.sin(self.theta1+tscale),np.cos(self.theta1+tscale)
        self.screws[1,:] = np.sin(self.theta2-tscale),np.cos(self.theta2-tscale)
        self.screws[2,:] = np.sin(self.theta1+tscale),np.cos(self.theta1+tscale)
        self.screws[3,:] = np.sin(self.theta2-tscale),np.cos(self.theta2-tscale)
        self.screws[0,:] *= (inrad+rscale)
        self.screws[1,:] *= (inrad+rscale)
        self.screws[2,:] *= (ourad-rscale)
        self.screws[3,:] *= (ourad-rscale)
              
        self.nsamp = 0
        self.values = []        
        
        if self.kind == "flexible":
            self.solve      = self._solve_flexi
            self.corr_point = self._corr_point_flexi
        elif self.kind == "rigid":
            self.solve      = self._solve_rigid
            self.corr_point = self._corr_point_rigid
        elif self.kind == "single":
            self.solve      = self._solve_single
            self.corr_point = self._corr_point_single
        elif self.kind == "xyparaboloid":
            self.solve       = self._solve_flexi_scipy
            self.corr_point  = self._corr_point_flexi_scipy
            self._paraboloid = self._xyaxes_paraboloid
        elif self.kind == "thetaparaboloid":
            self.solve       = self._solve_flexi_scipy
            self.corr_point  = self._corr_point_flexi_scipy
            self._paraboloid = self._rotated_paraboloid
        elif self.kind == "fixedtheta":
            self.solve       = self._solve_flexi_scipy
            self.corr_point  = self._corr_point_flexi_scipy
            self._paraboloid = self._fixed_paraboloid
        else:
            raise Exception("Unknown panel kind: ",self.kind)

        
    def is_inside(self,rad,phi):
        # Simple test of polar coordinates to check that a point is
        # inside this panel
        angle  = self.theta1 <= phi <= self.theta2
        radius = self.inrad  <= rad <= self.ourad
        return (angle and radius)

    
    def add_point(self,value):
        self.values.append(value)
        self.nsamp += 1


    def _solve_flexi(self):
        # Solve panel adjustments for flexible VLA style panels by
        # constructing a system of 4 linear equations
        syssize = 4
        if self.nsamp < syssize:
            # In this case the matrix will always be singular as the
            # rows will be linear combinations
            return
        system = np.zeros([syssize,syssize])
        vector = np.zeros(syssize)
        
        for ipoint in range(len(self.values)):
            dev = self.values[ipoint][-1]
            if dev != 0:
                xcoor = self.values[ipoint][0]
                ycoor = self.values[ipoint][1]
                fac   = self.bmp[0]+ycoor*(self.tmp[0]-self.bmp[0])/self.tmp[1]
                coef1 = (self.tmp[1]-ycoor) * (1.-xcoor/fac) / (2.0*self.tmp[1])
                coef2 = ycoor * (1.-xcoor/fac) / (2.0*self.tmp[1])
                coef3 = (self.tmp[1]-ycoor) * (1.+xcoor/fac) / (2.0*self.tmp[1])
                coef4 = ycoor * (1.+xcoor/fac) / (2.0*self.tmp[1])
                system[0,0] += coef1*coef1
                system[0,1] += coef1*coef2
                system[0,2] += coef1*coef3
                system[0,3] += coef1*coef4
                system[1,0]  = system[0,1]
                system[1,1] += coef2*coef2
                system[1,2] += coef2*coef3
                system[1,3] += coef2*coef4
                system[2,0]  = system[0,2]
                system[2,1]  = system[1,2]
                system[2,2] += coef3*coef3
                system[2,3] += coef3*coef4
                system[3,0]  = system[0,3]
                system[3,1]  = system[1,3]
                system[3,2]  = system[2,3]
                system[3,3] += coef4*coef4
                vector[0]   = vector[0] + dev*coef1
                vector[1]   = vector[1] + dev*coef2
                vector[2]   = vector[2] + dev*coef3
                vector[3]   = vector[3] + dev*coef4

        self.par = gauss_numpy(system,vector)
        self.solved = True
        return


    def _solve_flexi_scipy(self,verbose=False):
        devia = np.ndarray(self.nsamp)
        coords = np.ndarray([2,self.nsamp])
        for i in range(self.nsamp):
            devia[i] = self.values[i][-1]
            coords[:,i] = self.values[i][0],self.values[i][1]
        
        if self.kind == "thetaparaboloid":
            liminf = [0, 0, -np.inf, 0.0]
            limsup = [np.inf, np.inf, np.inf, np.pi]
            p0     = [1e2, 1e2, np.mean(devia), 0]
        elif self.kind == "xyparaboloid" or self.kind == "fixedtheta":
            liminf = [0, 0, -np.inf]
            limsup = [np.inf, np.inf, np.inf]
            p0     = [1e2, 1e2, np.mean(devia)]
            
        maxfevs=[1e5,1e6,1e7]
        for maxfev in maxfevs:
            try:
                popt, pcov = opt.curve_fit(self._paraboloid,coords,devia,
                                           p0=p0, bounds=[liminf,limsup],
                                           maxfev=maxfev)
            except RuntimeError:
                if (verbose):
                    print("Increasing number of iterations")
                continue
            else:
                if (verbose):
                    print("Converged with less than {0:d} iterations".format(maxfev))
                break
        self.par = popt
        self.solved = True

        
    def _fixed_paraboloid(self,coords,ucurv,vcurv,zoff):
        # Same as the rotated paraboloid, but theta is the panel center
        # This assumes that the center of the paraboloid is the center
        # of the panel, is this reasonable?
        # Also this function can produce degeneracies, due to the fact
        # that there are multiple combinations of theta ucurv and
        # vcurv that produce the same paraboloid
        x,y = coords
        xc,yc = self.center
        u = (x-xc)*np.cos(self.zeta) + (y-yc)*np.sin(self.zeta)
        v = (x-xc)*np.sin(self.zeta) + (y-yc)*np.cos(self.zeta)
        return -((u/ucurv)**2+(v/vcurv)**2)+zoff

    
    def _rotated_paraboloid(self,coords,ucurv,vcurv,zoff,theta):
        # This assumes that the center of the paraboloid is the center
        # of the panel, is this reasonable?
        # Also this function can produce degeneracies, due to the fact
        # that there are multiple combinations of theta ucurv and
        # vcurv that produce the same paraboloid
        x,y = coords
        xc,yc = self.center
        u = (x-xc)*np.cos(theta) + (y-yc)*np.sin(theta)
        v = (x-xc)*np.sin(theta) + (y-yc)*np.cos(theta)
        return -((u/ucurv)**2+(v/vcurv)**2)+zoff

    
    def _xyaxes_paraboloid(self,coords,a,b,c):
        # This assumes that the center of the paraboloid is the center
        # of the panel, is this reasonable?
        x,y = coords
        return -(((x-self.center[0])/a)**2+((y-self.center[1])/b)**2)+c
    
        
    def _solve_rigid(self):
        # Solve panel adjustments for rigid tilt and shift only panels by
        # constructing a system of 3 linear equations
        syssize = 3
        if self.nsamp < syssize:
            # In this case the matrix will always be singular as the
            # rows will be linear combinations
            return
        system = np.zeros([syssize,syssize])
        vector = np.zeros(syssize)
        for ipoint in range(len(self.values)):
            if self.values[ipoint][-1] != 0:
                system[0,0] += self.values[ipoint][0]*self.values[ipoint][0]
                system[0,1] += self.values[ipoint][0]*self.values[ipoint][1]
                system[0,2] += self.values[ipoint][0]
                system[1,0]  = system[0,1]
                system[1,1] += self.values[ipoint][1]*self.values[ipoint][1]
                system[1,2] += self.values[ipoint][1]
                system[2,0]  = system[0,2]
                system[2,1]  = system[1,2]
                system[2,2] += 1.0
                vector[0]   += self.values[ipoint][-1]*self.values[ipoint][0]
                vector[1]   += self.values[ipoint][-1]*self.values[ipoint][1]
                vector[2]   += self.values[ipoint][-1]
                
        self.par = gauss_numpy(system,vector)
        self.solved = True
        return

    
    def _solve_single(self):
        if self.nsamp > 0:
            # Solve panel adjustments for rigid vertical shift only panels
            self.par = np.zeros(1)
            shiftmean = 0.
            ncount    = 0
            for value in self.values:
                if value[-1] != 0:
                    shiftmean += value[-1]
                    ncount    += 1
                    
            shiftmean  /= ncount
            self.par[0] = shiftmean
            self.solved = True
        else:
            self.solved = False
        return


    def get_corrections(self):
        if not self.solved:
            raise Exception("Cannot correct a panel that is not solved")
        self.corr = np.ndarray(len(self.values))
        icorr = 0
        for val in self.values:
            self.corr[icorr] = self.corr_point(val[0],val[1])
            icorr+=1
        return          

    
    def _corr_point_flexi(self,xcoor,ycoor):
        coef = np.ndarray(4)
        corrval = 0
        fac   = self.bmp[0]+ycoor*(self.tmp[0]-self.bmp[0])/self.tmp[1]
        coef[0] = (self.tmp[1]-ycoor) * (1.-xcoor/fac) / (2.0*self.tmp[1])
        coef[1] = ycoor * (1.-xcoor/fac) / (2.0*self.tmp[1])
        coef[2] = (self.tmp[1]-ycoor) * (1.+xcoor/fac) / (2.0*self.tmp[1])
        coef[3] = ycoor * (1.+xcoor/fac) / (2.0*self.tmp[1])
        for ipar in range(len(self.par)):
            corrval += coef[ipar]*self.par[ipar]
        return corrval

    
    def _corr_point_flexi_scipy(self,xcoor,ycoor):
        corrval = self._paraboloid([xcoor,ycoor],*self.par)
        return corrval

    
    def _corr_point_rigid(self,xcoor,ycoor):
        return xcoor*self.par[0] + ycoor*self.par[1] + self.par[2]


    def _corr_point_single(self,xcoor,ycoor):
        return self.par[0]

    
    def export_adjustments(self, unit='mm', screen=False):
        if unit == 'mm':
            fac = 1.0
        elif unit == 'miliinches':
            fac = 1000.0/25.4
        else:
            raise Exception("Unknown unit: "+unit)
        
        string = '{0:8d} {1:8d}'.format(self.iring,self.ipanel)
        for screw in self.screws[:,]:
            string += ' {0:10.2f}'.format(fac*self.corr_point(*screw))
        if screen:
            print(string)
        return string


    def print_misc(self,verbose=False):
        print("########################################")
        print("{0:20s}={1:8d}".format("ipanel",self.ipanel))
        print("{0:20s}={1:8s}".format("kind"," "+self.kind))
        print("{0:20s}={1:8.5f}".format("inrad",self.inrad))
        print("{0:20s}={1:8.5f}".format("ourad",self.ourad))
        print("{0:20s}={1:8.5f}".format("theta1",self.theta1))
        print("{0:20s}={1:8.5f}".format("theta2",self.theta2))
        print("{0:20s}={1:8.5f}".format("zeta",self.zeta))
        print("{0:20s}={1:8.5f}, {2:8.5f}".format("bmp",*self.bmp))
        print("{0:20s}={1:8.5f}, {2:8.5f}".format("tmp",*self.tmp))
        print("{0:20s}={1:8d}".format("nsamp",self.nsamp))
        if verbose:
            for isamp in range(self.nsamp):
                strg = "{0:20s}=".format("samp{0:d}".format(isamp))
                for val in self.values[isamp]:
                    strg+= str(val)+", "
                print(strg)
        print()


    def plot(self,ax,screws=False):
        lw = 0.5
        msize = 2
        x1 = self.inrad*np.sin(self.theta1)
        y1 = self.inrad*np.cos(self.theta1)
        x2 = self.ourad*np.sin(self.theta1)
        y2 = self.ourad*np.cos(self.theta1)
        ax.plot([x1, x2],[y1, y2], ls='-',color='black',marker = None, lw=lw)
        scale = 0.05
        rt = (self.inrad+self.ourad)/2
        xt = rt*np.sin(self.zeta)
        yt = rt*np.cos(self.zeta)
        ax.text(xt,yt,str(self.ipanel),fontsize=5,ha='center')
        if screws:
            markers = ['x','o','*','+']
            colors  = ['g','g','r','r']
            for iscrew in range(self.screws.shape[0]):
                screw = self.screws[iscrew,]
                ax.scatter(screw[0],screw[1], marker=markers[iscrew],
                           lw=lw, s=msize, color=colors[iscrew] )
        if self.ipanel == 1:
            inrad = plt.Circle((0, 0), self.inrad, color='black', fill=False,lw=lw)
            ourad = plt.Circle((0, 0), self.ourad, color='black', fill=False,lw=lw)
            ax.add_patch(inrad)
            ax.add_patch(ourad)

            
class Antenna_Surface:
    # Describes the antenna surface properties, as well as being
    # capable of computing the gains and rms over the surface.
    # Heavily dependent on telescope architecture, Panel geometry to
    # be created by the specific _init_tel routine

    def __init__(self,amp,dev,telescope,cutoff=0.21,pkind=None):
        # Initializes antenna surface parameters    
        self.ampfile = amp
        self.devfile = dev
        
        self._read_images()        
        self.cut = cutoff*np.max(self.amp)
        print(self.cut)
        
        if telescope == 'VLA':
            self._init_vla()
        elif telescope == 'VLBA':
            self._init_vlba()
        else:
            raise Exception("Unknown telescope: "+telescope)
        if not pkind is None:
            self.panelkind = pkind

        self._get_aips_headpars()
        self.reso  = self.diam/self.npoint

        self.resi = None
        self.solved = False
        if self.ringed:
            self._build_polar()
            self._build_ring_panels()
            self._build_ring_mask()
            self.fetch_panel = self._fetch_panel_ringed
            self.compile_panel_points = self._compile_panel_points_ringed
            self.compile_panel_points_numba = self._compile_panel_points_ringed_numba
        

    def _get_aips_headpars(self):
        for line in self.devhead["HISTORY"]:
            wrds = line.split()
            if wrds[1] == "Visibilities":
                self.npoint = np.sqrt(int(wrds[-1]))
            elif wrds[1] == "Observing":
                # Stored in mm
                self.wavel = 1000*float(wrds[-2]) 
            elif wrds[1] == "Antenna" and wrds[2] == "surface":
                self.inlim = abs(float(wrds[-3]))
                self.oulim = abs(float(wrds[-2]))
        
        
    def _read_images(self):
        self.amphead,self.amp = read_fits(self.ampfile)
        self.devhead,self.dev = read_fits(self.devfile)
        self.dev *= 1000
        #
        if self.devhead['NAXIS1'] != self.amphead['NAXIS1']:
            raise Exception("Amplitude and deviation images have different sizes")
        self.npix = int(self.devhead['NAXIS1'])
        self.xaxis = Linear_Axis(self.npix,self.amphead["CRPIX1"],
                                 self.amphead["CRVAL1"],self.amphead["CDELT1"])
        self.yaxis = Linear_Axis(self.npix,self.amphead["CRPIX2"],
                                 self.amphead["CRVAL2"],self.amphead["CDELT2"])
        return

    
    def _build_ring_mask(self):
        self.mask = np.where(self.amp < self.cut, False, True)
        self.mask = np.where(self.rad > self.inlim, self.mask, False)
        self.mask = np.where(self.rad < self.oulim, self.mask, False)
        self.mask = np.where(np.isnan(self.dev), False, self.mask)

        
    # Other known telescopes should be included here, ALMA, ngVLA
    def _init_vla(self):
        # Initializes surfaces according to VLA antenna parameters
        self.panelkind = 'flexible'
        self.telescope = "VLA"
        self.diam      = 25.0  # meters
        self.focus     = 8.8   # meters
        self.ringed    = True
        self.nrings    = 6
        self.npanel    = [12,16,24,40,40,40]
        self.inrad     = [1.983, 3.683, 5.563, 7.391, 9.144, 10.87]
        self.ourad     = [3.683, 5.563, 7.391, 9.144, 10.87, 12.5 ]
        self.inlim     = 2.0
        self.oulim     = 12.0

        
    def _init_vlba(self):
        # Initializes surfaces according to VLBA antenna parameters
        self.panelkind = "flexible"
        self.telescope = "VLBA"
        self.diam      = 25.0  # meters
        self.focus     = 8.75  # meters
        self.ringed    = True
        self.nrings    = 6
        self.npanel    = [20,20,40,40,40,40]
        self.inrad     = [1.676,3.518,5.423,7.277, 9.081,10.808]
        self.ourad     = [3.518,5.423,7.277,9.081,10.808,12.500]
        self.inlim     = 2.0
        self.oulim     = 12.0


    def _build_polar(self):
        self.rad = np.zeros([self.npix,self.npix])
        self.phi = np.zeros([self.npix,self.npix])
        for iy in range(self.npix):
            ycoor = self.yaxis.idx_to_coor(iy+0.5)
            for ix in range(self.npix):
                xcoor = self.xaxis.idx_to_coor(ix+0.5)
                self.rad[ix,iy] = np.sqrt(xcoor**2+ycoor**2)
                self.phi[ix,iy] = np.arctan2(ycoor,xcoor)
                if self.phi[ix,iy]<0:
                    self.phi[ix,iy] += 2*np.pi


    def _build_ring_panels(self):
        self.panels = []
        for iring in range(self.nrings):
            angle = 2.0*np.pi/self.npanel[iring]
            for ipanel in range(self.npanel[iring]):
                panel = Ring_Panel(self.panelkind, angle, iring,
                                   ipanel, self.inrad[iring], self.ourad[iring])
                self.panels.append(panel)
        return


    def _compile_panel_points_ringed(self):
        for iy in range(self.npix):
            yc = self.yaxis.idx_to_coor(iy+0.5)
            for ix in range(self.npix):
                if self.mask[ix,iy]:
                    xc = self.xaxis.idx_to_coor(ix+0.5)
                    # How to do the coordinate choice here without
                    # adding an if?
                    for panel in self.panels:
                        if panel.is_inside(self.rad[ix,iy],self.phi[ix,iy]):
                            panel.add_point([xc,yc,ix,iy,self.dev[ix,iy]])

    @jit
    def _compile_panel_points_ringed_numba(self):
        for iy in range(self.npix):
            yc = self.yaxis.idx_to_coor(iy+0.5)
            for ix in range(self.npix):
                if self.mask[ix,iy]:
                    xc = self.xaxis.idx_to_coor(ix+0.5)
                    # How to do the coordinate choice here without
                    # adding an if?
                    for panel in self.panels:
                        if panel.is_inside(self.rad[ix,iy],self.phi[ix,iy]):
                            panel.add_point([xc,yc,ix,iy,self.dev[ix,iy]])

                            
    def _fetch_panel_ringed(self,ring,panel):
        if ring == 1:
            ipanel = panel-1
        else:
            ipanel = np.sum(self.npanel[:ring-1])+panel-1
        return self.panels[ipanel]

    
    def gains(self):
        self.ingains = self._gains_array(self.dev)
        if not self.resi is None:
            self.ougains = self._gains_array(self.resi)
            return self.ingains, self.ougains
        else:
            return self.ingains

        
    def _gains_array(self,arr):
        # Compute the actual and theoretical gains for the current
        # antenna surface. What is the unit for the wavelength? mm 
        forpi = 4.0*np.pi
        fact = 1000. * self.reso / self.wavel
        fact *= fact
        #
        # What are these sums?
        sumrad   = 0.0
        sumtheta = 0.0
        nsamp    = 0
        #    convert surface error to phase
        #    and compute gain loss
        for iy in range(self.npix): 
            for ix in range(self.npix):
                if self.mask[ix,iy]:
                    quo = self.rad[ix,iy] / (2.*self.focus)
                    phase     = arr[ix,iy]*forpi/(np.sqrt(1.+quo*quo)*self.wavel)
                    sumrad   += np.cos(phase)
                    sumtheta += np.sin(phase)
                    nsamp    += 1              

        ampmax  = np.sqrt(sumrad*sumrad + sumtheta*sumtheta)
        if (nsamp<=0):
            raise Exception("Antenna is blanked")
        ampmax *= fact/nsamp
        gain    = ampmax*forpi
        thgain  = fact*forpi
        #
        gain    = convert_to_db(gain)
        thgain  = convert_to_db(thgain)
        return gain,thgain

    
    def get_rms(self):
        # Compute the RMS of the antenna surface
        self.inrms = np.sqrt(np.mean(self.dev[self.mask]**2))
        if not self.resi is None:
            self.ourms = np.sqrt(np.mean(self.resi[self.mask]**2))
            return self.inrms, self.ourms
        else:
            return self.inrms

    
    def fit_surface(self):
        for panel in self.panels:
            panel.solve()
        self.solved = True


    def correct_surface(self):
        if not self.solved:
            raise Exception("Panels must be fitted before atempting a correction")
        self.corr = np.where(self.mask,0,np.nan)
        self.resi = np.copy(self.dev)
        for panel in self.panels:
            panel.get_corrections()
            for ipnt in range(len(panel.corr)):
                val = panel.values[ipnt]
                ix,iy = int(val[2]),int(val[3])
                self.resi[ix,iy] -=  panel.corr[ipnt]
                self.corr[ix,iy]  = -panel.corr[ipnt]

                
    def print_misc(self):
        for panel in self.panels:
            panel.print_misc()


    def plot_surface(self,filename=None,mask=False,screws=False):
        vmin,vmax = np.nanmin(self.dev),np.nanmax(self.dev)
        rms = self.get_rms()
        if mask:
            fig, ax = plt.subplots(1,2,figsize=[10,5])
            title = "Mask"
            self._plot_surface(self.mask,title,fig,ax[0],0,1,screws=screws,
                               mask=mask)
            vmin,vmax = np.nanmin(self.amp),np.nanmax(self.amp)
            title = "Amplitude min={0:.5f}, max ={1:.5f}".format(vmin,vmax)
            self._plot_surface(self.amp,title,fig,ax[1],vmin,vmax,screws=screws,
                               unit=self.amphead["BUNIT"].strip())
        else:
            if self.resi is None:
                fig, ax = plt.subplots()
                title = "Before correction\nRMS = {0:8.5} mm".format(rms)
                self._plot_surface(self.dev,title,fig,ax,vmin,vmax,screws=screws)
            else:
                fig, ax = plt.subplots(1,3,figsize=[15,5])
                title = "Before correction\nRMS = {0:.3} mm".format(rms[0])
                self._plot_surface(self.dev,title,fig,ax[0],vmin,vmax,screws=screws)
                title = "Corrections"
                self._plot_surface(self.corr,title,fig,ax[1],vmin,vmax,screws=screws)
                title = "After correction\nRMS = {0:.3} mm".format(rms[1])
                self._plot_surface(self.resi,title,fig,ax[2],vmin,vmax,screws=screws)
        fig.suptitle("Antenna Surface")
        fig.tight_layout()
        if (filename is None):
            plt.show()
        else:
            plt.savefig(filename, dpi=600)

            
    def _plot_surface(self,data,title,fig,ax,vmin,vmax,screws=False,mask=False,
                      unit='mm'):
        ax.set_title(title)
        # set the limits of the plot to the limits of the data
        xmin = self.xaxis.idx_to_coor(-0.5)
        xmax = self.xaxis.idx_to_coor(self.xaxis.n-0.5)
        ymin = self.yaxis.idx_to_coor(-0.5)
        ymax = self.yaxis.idx_to_coor(self.yaxis.n-0.5)
        im   = ax.imshow(np.flipud(data), cmap='viridis', interpolation='nearest',
                         extent=[xmin,xmax,ymin,ymax], vmin=vmin,vmax=vmax)
        divider = make_axes_locatable(ax)
        if not mask:
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, label="Z Scale ["+unit+"]", cax=cax)    
        ax.set_xlabel("X axis [m]")
        ax.set_ylabel("Y axis [m]")
        for panel in self.panels:
            panel.plot(ax,screws=screws)


    def export_corrected(self,filename):
        if self.resi is None:
            raise Exception("Cannot export corrected surface")
        hdu = fits.PrimaryHDU(self.resi)
        hdu.header = self.devhead
        hdu.header["ORIGIN"] = 'Astrohack PANEL'
        hdu.writeto(filename, overwrite=True)
        return


    def export_screw_adjustments(self,filename,unit='mm'):
        spc = ' '
        outfile = 'Screw adjustments for {0:s} {1:s} antenna\n'.format(
            self.telescope, self.amphead['telescop'])
        outfile += 'Adjustments are in '+unit+lnbr
        outfile += 2*lnbr
        outfile += 25*spc+"{0:22s}{1:22s}".format('Inner Edge','Outer Edge')+lnbr
        outfile += 5*spc+"{0:8s}{1:8s}".format("Ring","panel")
        outfile += 2*spc+2*"{0:11s}{1:11s}".format('left','right')+lnbr
        for panel in self.panels:
            outfile += panel.export_adjustments(unit=unit)+lnbr
        lefile = open(filename,'w')
        lefile.write(outfile)
        lefile.close()
            
            

