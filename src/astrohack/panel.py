import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits

# static methods not linked to any specific class
def gauss_solver(system):
    # This is actually a simple matrix inversion
    # can be replaced by a simple call to np.linalg.inv
    shape = system.shape
    if shape[0] != shape[1]:
        raise Exception("Matrix is not square")
    size = shape[0]
    pivot = np.zeros(size)
    rowidx = np.zeros(size)
    colidx = np.zeros(size)

    # Main diagonalization Loop
    for iidx in range(size):
        big = 0.
        for jidx in range(size):
            if pivot[jidx] != 1:
                for kidx in range(size):
                    if pivot[kidx] == 0:
                        if abs(system[jidx,kidx]) >= big:
                            big  = abs(system[jidx,kidx])
                            irow = jidx
                            icol = kidx
                    elif pivot[kidx] > 1:
                        raise Exception("matrix is singular")

        pivot[icol] += 1
        if irow != icol:
            for jidx in range(size):
                copy = system[irow,jidx]
                system[irow,jidx] = system[icol,jidx]
                system[icol,jidx] = copy

        rowidx[iidx] = irow
        colidx[iidx] = icol
        if system[icol,icol] == 0:
            raise Exception("matrix is singular")
            
        pivinv = 1.0/system[icol,icol]
        system[icol,:] *= pivinv

        for jidx in range(size):
            if jidx != icol:
                copy = system[jidx,icol]
                system[jidx,icol] = 0.
                system[jidx,:] -= system[icol,:]*copy
    
    # Final test on diagonalization
    for iidx in range(size-1,-1,-1):
        if rowidx[iidx] != colidx[iidx]:
            for kidx in range(size):
                copy = system[kidx,rowidx[iidx]]
                system[kidx,rowidx[iidx]] = system[kidx,colidx[iidx]]
                system[kidx,colidx[iidx]] = copy

    return system


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
        data = hdul[0].data[0,0,:,:]*1000
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

    
class Panel:
    # Main working class that defines and works on panel objects
    # Currently this class strongly relies on the fact that panels are
    # sections of a ring of equal angular size, inner and outer radii
    
    def __init__(self,kind,angle,ipanel,inrad,ourad,bmp,tmp):
        # Panel initialization
        self.kind   = kind
        self.ipanel = ipanel
        self.inrad  = inrad
        self.ourad  = ourad
        self.theta1 = ipanel*angle
        self.theta2 = (ipanel+1)*angle
        self.zeta   = (ipanel+0.5)*angle
        self.bmp    = bmp
        self.tmp    = tmp
        self.built  = False
        self.solved = False

        
    def is_inside(self,rad,phi):
        # Simple test of polar coordinates to check that a point is
        # inside this panel
        angle  = self.theta1 <= phi <= self.theta2
        radius = self.inrad  <= rad <= self.ourad
        return (angle and radius)

    
    def compute_points(self,amp,dev,xaxis,yaxis):
        # Most computationally intensive routine so far.  This
        # routines loops over points potentially inside a panel and
        # checks wather they are inside the panel or not. If point is
        # inside the panel store its x and y coordinates, its ix and
        # iy indexes as well as its deviation
        inc = 15.0 / amp.shape[0]
        nsamp = 0
        ycoor = self.bmp[1]
        values = []
        # Loop over potential y coordinates
        while (ycoor<=self.tmp[1]):
            deltax = self.bmp[0]+(self.tmp[0]-self.bmp[0])*(ycoor-self.bmp[1])/(self.tmp[1]-self.bmp[1])
            xcoor = -deltax
            # loop over potential x coordinates
            while (xcoor<=deltax):
                phi1 = np.arctan2 (xcoor, ycoor)
                phipoint = phi1 + self.zeta
                radpoint = np.sqrt(xcoor**2 + ycoor**2)
                xpoint = radpoint * np.sin(phipoint)
                ypoint = radpoint * np.cos(phipoint)
                xint = round(xaxis.coor_to_idx(xpoint))
                yint = round(yaxis.coor_to_idx(ypoint))
                #    checks :
                #    a. is this point truly
                #    within the panel?
                xpoint = xaxis.idx_to_coor(xint)
                ypoint = yaxis.idx_to_coor(xint)
                radpoint = np.sqrt(xpoint**2 + ypoint**2)
                phipoint = np.arctan2 (xpoint, ypoint)
                while (phipoint<0.):
                    phipoint += 2.0*np.pi
                #    b.  have we used this point
                if (amp[xint,yint]>0 and self.is_inside(radpoint,phipoint)):
                    if (dev[xint,yint] != 0.0):
                        nsamp += 1
                        #    set up the parameters for the
                        #    panel solution -- locate each
                        #    point on the reference panel
                        phi1 = phipoint - self.zeta
                        value = [radpoint*np.sin(phi1),
                                 radpoint*np.cos(phi1)-self.bmp[1],
                                 xint,yint,
                                 dev[xint,yint]]
                        values.append(value)
                        
                xcoor = xcoor + inc
                
            ycoor = ycoor + inc

        self.nsamp  = nsamp
        self.values = values
        self.built  = True
        return


    def compute_points_new(self,amp,dev,rad,phi):
        self.nsamp = 0
        self.values = []
        for iy in range(rad.shape[0]):
            for ix in range(rad.shape[0]):
                if self.is_inside(rad[ix,iy],phi[ix,iy]): 
                    if dev[ix,iy] > 0 and amp[ix,iy] > 0:
                        self.nsamp += 1
                        value = [rad[ix,iy]*np.sin(phi[ix,iy]),
                                 rad[ix,iy]*np.cos(phi[ix,iy])-self.bmp[1],
                                 ix,iy,
                                 dev[ix,iy]]
                        self.values.append(value)
        self.built = True
        return

    
    def solve(self):
        if not self.built:
            raise Exception("Cannot solve a panel that is not built")
        # Calls apropriated solving function based on panel type The
        # parameters computed here are to be used in the future to
        # compute the actual screw adjustments.
        #
        try:
            if self.kind == "flexible":
                self._solve_flexi()
            elif self.kind == "rigid":
                self._solve_rigid()
            elif self.kind == "single":
                self._solve_single()
            else:
                raise Exception("Don't know how to solve panel of kind: ",self.kind)
        except:
            self.print_misc(verbose=True)
        return

    
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
        # Solve panel adjustments for rigid vertical shift only panels
        self.par = np.zeros(1)
        shiftmean = 0.
        ncount    = 0
        for ipoint in range(len(self.values)):
            if self.values[ipoint][-1] != 0:
                shiftmean += self.values[ipoint][-1]
                ncount    += 1

        shiftmean  /= ncount
        self.par[0] = shiftmean
        self.solved = True
        return


    def get_corrections(self):
        if not self.solved:
            raise Exception("Cannot correct a panel that is not solved")
        # Flexible VLA like panel
        if self.kind == "flexible":
            self._get_corr_flexi()
        # Rigid but can be tilted
        elif self.kind == "rigid":
            self._get_corr_rigid()
        # Rigid and can only be moved vertically
        elif self.kind == "single":
            self._get_corr_single()
        else:
            raise Exception("Don't know how to solve panel of kind: ",self.kind)

        
    def _get_corr_flexi(self):
        self.corr = np.ndarray(len(self.values))
        coef = np.ndarray(4)
        icorr = 0
        for val in self.values:
            xcoor,ycoor = val[0:2]
            corrval = 0
            fac   = self.bmp[0]+ycoor*(self.tmp[0]-self.bmp[0])/self.tmp[1]
            coef[0] = (self.tmp[1]-ycoor) * (1.-xcoor/fac) / (2.0*self.tmp[1])
            coef[1] = ycoor * (1.-xcoor/fac) / (2.0*self.tmp[1])
            coef[2] = (self.tmp[1]-ycoor) * (1.+xcoor/fac) / (2.0*self.tmp[1])
            coef[3] = ycoor * (1.+xcoor/fac) / (2.0*self.tmp[1])
            for ipar in range(len(self.par)):
                corrval += coef[ipar]*self.par[ipar]
            self.corr[icorr] = corrval
            icorr+=1
        return


    def _get_corr_rigid(self):
        self.corr = np.ndarray(len(self.values))
        icorr = 0
        for val in self.values:
            corrval = val[0]*self.par[0] + val[1]*self.par[1] + self.par[2]
            self.corr[icorr] = corrval
            icorr+=1
        return


    def _get_corr_single(self):
        self.corr = np.ndarray(len(self.values))
        icorr = 0
        for val in self.values:
            corrval = self.par[0]
            self.corr[icorr] = corrval
            icorr+=1
        return

    
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
        if (self.built):
            print("{0:20s}={1:8d}".format("nsamp",self.nsamp))
            if verbose:
                for isamp in range(self.nsamp):
                    strg = "{0:20s}=".format("samp{0:d}".format(isamp))
                    for val in self.values[isamp]:
                        strg+= str(val)+", "
                    print(strg)
        print()
        
        
class Ring:
    # Class created just for hierarchical pourposes, irrelevant if
    # dish is not circular or panel design is not organized in rings
    
    def __init__(self,kind,npanel,inrad,ourad):
        # Ring initialization
        self.kind   = kind
        self.npanel = npanel
        self.inrad  = inrad
        self.ourad  = ourad
        self.angle  = 2.0*np.pi/npanel
        self.bmp = [inrad*np.sin(self.angle/2.0),inrad*np.cos(self.angle/2.0)]
        self.tmp = [ourad*np.sin(self.angle/2.0),ourad*np.cos(self.angle/2.0)]


    def create_panels(self,amp,dev,xaxis,yaxis):
        # Creates and computes the point inside each panel of the ring
        self.panels = []
        for ipanel in range(self.npanel):
            panel = Panel(self.kind,self.angle,ipanel,self.inrad,self.ourad,
                          self.bmp,self.tmp)
            panel.compute_points(amp,dev,xaxis,yaxis)
            self.panels.append(panel)

            
    def create_panels_new(self,amp,dev,rad,phi):
        # Creates and computes the point inside each panel of the ring
        self.panels = []
        for ipanel in range(self.npanel):
            panel = Panel(self.kind,self.angle,ipanel,self.inrad,self.ourad,
                          self.bmp,self.tmp)
            panel.compute_points_new(amp,dev,rad,phi)
            self.panels.append(panel)

            
    def solve_panels(self):
        # Calls the panels to solve their adjustments
        for ipanel in range(self.npanel):
            self.panels[ipanel].solve()
            

    def print_misc(self):
        if not (self.panels is None):
            for panel in self.panels:
                panel.print_misc()
                print()


class Antenna_Surface:
    # Describes the antenna surface properties, as well as being
    # capable of computing the gains and rms over the surface.
    # Heavily dependent on telescope architecture, currently only
    # telescopes that have panels arranged in rings can be modeled
    # here.

    def __init__(self,amp,dev,npoint,telescope):
        # Initializes antenna surface parameters
        if telescope == 'VLA':
            self._init_vla()
        elif telescope == 'VLBA':
            self._init_vlba()
        else:
            raise Exception("VLA is the only know telescope for the moment")
        
        self.ampfile = amp
        self.devfile = dev
        self._read_images()
        
        self.rms   = np.nan
        # Is this really how to compute this?
        self.reso  = self.diam/npoint

        ref = -1.0
        inc = self.diam/self.npix
        val = -self.diam/2.-inc/2
        self.xaxis = Linear_Axis(self.npix,ref,val,inc)
        self.yaxis = Linear_Axis(self.npix,ref,val,inc)
        self._build_polar()


        
    def _read_images(self):
        self.amphead,self.amp = read_fits(self.ampfile)
        self.devhead,self.dev = read_fits(self.devfile)
        #
        if self.devhead['NAXIS1'] != self.amphead['NAXIS1']:
            raise Exception("Amplitude and deviation images have different sizes")
        self.npix = int(self.devhead['NAXIS1'])
        return
    
        
    # Other known telescopes should be included here, ALMA, ngVLA
    def _init_vla(self):
        # Initializes surfaces according to VLA antenna parameters
        self.panelkind = "flexible"
        # self.panelkind = "rigid"
        # self.panelkind = "single"
        self.telescope = "VLA"
        self.diam      = 25.0  # meters
        self.focus     = 8.8   # meters
        self.nrings    = 6
        self.npanel    = [12,16,24,40,40,40]
        self.inrad     = [1.983, 3.683, 5.563, 7.391, 9.144, 10.87]
        self.ourad     = [3.683, 5.563, 7.391, 9.144, 10.87, 12.5 ]

        
    def _init_vlba(self):
        # Initializes surfaces according to VLBA antenna parameters
        self.panelkind = "unknown"
        self.telescope = "VLBA"
        self.diam      = 25.0  # meters
        self.focus     = 8.75  # meters
        self.nrings    = 6
        self.npanel    = [20,20,40,40,40,40]
        self.inrad     = [1.676,3.518,5.423,7.277, 9.081,10.808]
        self.ourad     = [3.518,5.423,7.277,9.081,10.808,12.500]


    def _build_polar(self):
        self.rad = np.zeros([self.npix,self.npix])
        self.phi = np.zeros([self.npix,self.npix])
        for iy in range(self.npix):
            ycoor = self.yaxis.idx_to_coor(iy)
            for ix in range(self.npix):
                xcoor = self.xaxis.idx_to_coor(ix)
                self.rad[ix,iy] = np.sqrt(xcoor**2+ycoor**2)
                self.phi[ix,iy] = np.arctan2(xcoor,ycoor)
                if self.phi[ix,iy]<0:
                    self.phi[ix,iy] +=2*np.pi


    def build_panels(self):
        # Loops over rings so rings can initialize and compute the
        # points inside their panels
        self.rings = []
        for iring in range(self.nrings):
            ring = Ring(self.panelkind,self.npanel[iring],self.inrad[iring],self.ourad[iring])
            ring.create_panels(self.amp,self.dev,self.xaxis,self.yaxis)
            self.rings.append(ring)
        return

        
    def build_panels_new(self):
        # Loops over rings so rings can initialize and compute the
        # points inside their panels
        self.rings = []
        for iring in range(self.nrings):
            ring = Ring(self.panelkind,self.npanel[iring],self.inrad[iring],self.ourad[iring])
            ring.create_panels_new(self.amp,self.dev,self.rad,self.phi)
            self.rings.append(ring)
        return

        
    def gains(self,wavel):
        # Compute the actual and theoretical gains for the current
        # antenna surface. What is the unit for the wavelength, cm or mm?
        forpi = 4.0*np.pi
        fact = 1000. * self.reso / wavel
        fact *= fact
        #
        # What are these sums?
        sumrad   = 0.0
        sumtheta = 0.0
        nsamp    = 0
        #    convert surface error to phase
        #    and compute gain loss
        for iy in range(self.dev.shape[1]): 
            for ix in range(self.dev.shape[0]):
                if self.amp[ix,iy]>0 and self.rad[ix,iy]<self.diam/2 and not np.isnan(self.dev[ix,iy]):
                    quo = self.rad[ix,iy] / (2.*self.focus)
                    phase     = self.dev[ix,iy]*forpi/(np.sqrt(1.+quo*quo)*wavel)
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

    
    def get_rms (self):
        # Compute the RMS of the antenna surface
        rms   = 0.0
        nsamp = 0.0
        for iy in range(self.dev.shape[1]): 
            for ix in range(self.dev.shape[0]):
                if self.amp[ix,iy]>0 and self.rad[ix,iy]<self.diam/2 and not np.isnan(self.dev[ix,iy]):
                    rms   += self.dev[ix,iy]**2
                    nsamp += 1
                    
        if (nsamp<=0):
            raise Exception("Antenna is blanked")
        rms = np.sqrt(rms/nsamp)
        self.rms = rms
        return rms

    
    def fit_adjustments(self):
        # loops over the rings so that they can loop over the panels
        # to compute the adjustments needed
        for iring in range(self.nrings):
            self.rings[iring].solve_panels()
        return


    def correct_surface(self):
        npoint = self.diam/self.reso
        corrected = Antenna_Surface(self.ampfile,self.devfile,npoint,self.telescope)
        iring = 0
        for ring in self.rings:
            iring += 1
            for panel in ring.panels:
                if (panel.solved):
                    panel.get_corrections()
                    nbad = 0
                    for ipnt in range(len(panel.corr)):
                        if panel.corr[ipnt] > 10:
                            nbad +=1
                        val = panel.values[ipnt]
                        ix,iy = int(val[0]),int(val[1])
                        corrected.dev[ix,iy] -= panel.corr[ipnt]
                    if nbad > 0:
                        print("**************************************************")
                        print("ring :",iring)
                        print("panel:",panel.ipanel)
                        print("nbad :",nbad)
                        print("nsamp:",panel.nsamp)
                    
        return corrected

    
    def print_misc(self):
        iring = 0
        for ring in self.rings:
            iring +=1
            print("************************************************************")
            print("ring: ",str(iring))
            ring.print_misc()
            print()


    def plot_deviations(self,panels=True):
        fig, ax = plt.subplots()
        ax.set_title('Antenna Surface')
        # set the limits of the plot to the limits of the data
        # xmin = self.xaxis.idx_to_coor(0)-self.xaxis.inc/2
        # xmax = self.xaxis.idx_to_coor(self.npix-1)+self.xaxis.inc/2
        # ymin = self.yaxis.idx_to_coor(0)-self.yaxis.inc/2
        # ymax = self.yaxis.idx_to_coor(self.npix-1)-self.yaxis.inc/2
        # ax.axis([xmin, xmax, ymin, ymax])
        plt.imshow(self.dev, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.show()
        
