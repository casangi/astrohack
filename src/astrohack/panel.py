import numpy as np


def gauss_solver(system):
    # Diagonalizes the system and then returns the diagonalized
    # version of it
    shape = system.shape
    if shape[0] != shape[1]:
        raise Exception("Matrix is not square")
    size = shape[0]
    pivot = np.zeros(size)
    rowidx = np.zeros(size)
    colidx = np.zeros(size)

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
    

    for iidx in range(size,0,-1):
        if rowidx[iidx] != colidx[iidx]:
            for kidx in range(size):
                copy = system[kidx,rowidx[iidx]]
                system[kidx,rowidx[iidx]] = system[kidx,colidx[iidx]]
                system[kidx,colidx[iidx]] = copy

    return system


def convert_to_db(val):
    return 10.*np.log10(val)

class Linear_Axis:
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
    def __init__(self,kind,angle,ipanel,inrad,ourad,blc,tlc):
        self.kind   = kind
        self.ipanel = ipanel
        self.inrad  = inrad
        self.ourad  = ourad
        self.theta1 = ipanel*angle
        self.theta2 = (ipanel+1)*angle
        self.zeta   = (ipanel+0.5)*angle
        self.blc    = blc
        self.tlc    = tlc

    def is_inside(self,rad,phi):
        angle  = self.theta1 <= phi <= self.theta2
        radius = self.inrad  <= rad <= self.ourad
        return (angle and radius)

    def compute_points(self,amp,dev,xaxis,yaxis):
        inc = 15.0 / amp.shape[0]
        nsamp = 0
        ycoor = self.blc[1]
        values = []
        while (ycoor<=self.tlc[1]):
            deltax = self.blc[0]+(self.tlc[0]-self.blc[0])*(ycoor-self.blc[1])/(self.tlc[1]-self.blc[1])
            xcoor = -deltax
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
                                 radpoint*np.cos(phi1)-self.blc[1],
                                 xint,yint,
                                 dev[xint,yint]]
                        values.append(value)
                        
                xcoor = xcoor + inc
                
            ycoor = ycoor + inc

        self.nsamp = nsamp
        self.values = values
        return

    def solve(self):
        # Flexible VLA like panel
        if self.kind == "flexible":
            self._solve_flexi()
        # Rigid but can be tilted
        elif self.kind == "rigid":
            self._solve_rigid()
        # Rigid and can only be moved vertically
        elif self.kind == "single":
            self._solve_single()
        else:
            raise Exception("Don't know how to solve panel of kind: ",self.kind)
        return

    def _solve_flexi(self):
        syssize = 4
        system = np.zeros([syssize,syssize])
        vector = np.zeros(syssize)
        
        for ipoint in range(len(self.values)):
            dev = self.values[ipoint][-1]
            if dev != 0:
                xcoor = self.values[ipoint][0]
                ycoor = self.values[ipoint][1]
                fac   = self.blc[0]+ycoor*(self.tlc[0]-self.blc[0])/self.tlc[1]
                coef1 = (self.tlc[2]-ycoor) * (1.-xcoor/fac) / (2.0*self.tlc[2])
                coef2 = ycoor * (1.-xcoor/fac) / (2.0*self.tlc[2])
                coef3 = (self.tlc[2]-ycoor) * (1.+xcoor/fac) / (2.0*self.tlc[2])
                coef4 = ycoor * (1.+xcoor/fac) / (2.0*self.tlc[2])
                system[1,1] += coef1*coef1
                system[1,2] += coef1*coef2
                system[1,3] += coef1*coef3
                system[1,4] += coef1*coef4
                system[2,1]  = system[1,2]
                system[2,2] += coef2*coef2
                system[2,3] += coef2*coef3
                system[2,4] += coef2*coef4
                system[3,1]  = system[1,3]
                system[3,2]  = system[2,3]
                system[3,3] += coef3*coef3
                system[3,4] += coef3*coef4
                system[4,1]  = system[1,4]
                system[4,2]  = system[2,4]
                system[4,3]  = system[3,4]
                system[4,4] += coef4*coef4
                vector[1]   = vector[1] + dev*a1
                vector[2]   = vector[2] + dev*a2
                vector[3]   = vector[3] + dev*a3
                vector[4]   = vector[4] + dev*a4

        newsys   = gauss_solver(system)
        self.par = np.zeros(syssize)
        for iidx in range(syssize):
            valsum = 0.
            for jidx in range(syssize):
                self.par[jidx] +=  newsys[jidx,iidx] * vector[jidx]
        return


    def _solve_rigid(self):
        syssize = 3
        system = np.zeros([syssize,syssize])
        vector = np.zeros(syssize)
        for ipoint in range(len(self.values)):
            if self.values[ipoint][-1] != 0:
                system[1,1] += self.values[ipoint][0]*self.values[ipoint][0]
                system[1,2] += self.values[ipoint][0]*self.values[ipoint][1]
                system[1,3] += self.values[ipoint][0]
                system[2,1]  = system[1,2]
                system[2,2] += self.values[ipoint][1]*self.values[ipoint][1]
                system[2,3] += self.values[ipoint][1]
                system[3,1]  = system[1,3]
                system[3,2]  = system[2,3]
                system[3,3] += 1.0
                vector[1]   += self.values[ipoint][-1]*self.values[ipoint][0]
                vector[2]   += self.values[ipoint][-1]*self.values[ipoint][1]
                vector[3]   += self.values[ipoint][-1]

        # Call the gauss seidel solver
        newsys   = gauss_solver(system)
        self.par = np.zeros(syssize)
        for iidx in range(syssize):
            valsum = 0.
            for jidx in range(syssize):
                self.par[jidx] +=  newsys[jidx,iidx] * vector[jidx]

    def _solve_single(self):
        self.par = np.zeros(1)
        shiftmean = 0.
        ncount    = 0
        for ipoint in range(len(self.values)):
            if self.values[ipoint][-1] != 0:
                shiftmean += self.values[ipoint][-1]
                ncount    += 1

        shiftmean  /= ncount
        self.par[0] = shiftmean
        
class Ring:
    def __init__(self,kind,npanel,inrad,ourad):
        self.kind   = kind
        self.npanel = npanel
        self.inrad  = inrad
        self.ourad  = ourad
        self.angle  = 2.0*np.pi/npanel
        self.blc = [inrad*np.sin(self.angle/2.0),inrad*np.cos(self.angle/2.0)]
        self.tlc = [ourad*np.sin(self.angle/2.0),ourad*np.cos(self.angle/2.0)]

    def create_panels(self,amp,dev,xaxis,yaxis):
        self.panels = []
        for ipanel in range(self.npanel):
            panel = Panel(self.kind,self.angle,ipanel,self.inrad,self.ourad,
                          self.blc,self.tlc)
            panel.compute_points(amp,dev,xaxis,yaxis)
            self.panels.append(panel)
        
    def solve_panels(self):
        for ipanel in range(self.npanel):
            self.panels[ipanel].solve()
    
class Antenna_Surface:

    def __init__(self,npoint,npix,telescope,perfect=False):
        if telescope == 'VLA':
            self._init_vla()
        elif telescope == 'VLBA':
            self._init_vlba()
        else:
            raise Exception("VLA is the only know telescope for the moment")

        self.npix  = npix  # pix
        self.rms   = np.nan
        # Is this really how to compute this?
        self.reso  = self.diam/npoint

        ref = -1.0
        inc = self.diam/self.npix
        val = -self.diam/2.-inc/2
        self.xaxis = Linear_Axis(self.npix,ref,val,inc)
        self.yaxis = Linear_Axis(self.npix,ref,val,inc)

        # Deviation map, for the moment created ad hoc, shall be read
        # from a file
        if perfect:
            self.dev  = np.zeros([npix,npix])
        else:
            # for the moment random to simulate a deformed antenna
            self.dev  = np.random.rand(npix,npix)-0.5
            
        # This amplitude image is used as a mask in subsequent
        # calculations
        self.amp = np.full([npix,npix],1.0)

    def _init_vla(self):
        self.panelkind = "single"
        self.telescope = "VLA"
        self.diam      = 25.0  # meters
        self.focus     = 8.8   # meters
        self.nrings    = 6
        self.npanel    = [12,16,24,40,40,40]
        self.inrad     = [1.983, 3.683, 5.563, 7.391, 9.144, 10.87]
        self.ourad     = [3.683, 5.563, 7.391, 9.144, 10.87, 12.5 ]

    def _init_vlba(self):
        self.panelkind = "unknown"
        self.telescope = "VLBA"
        self.diam      = 25.0  # meters
        self.focus     = 8.75  # meters
        self.nrings    = 6
        self.npanel    = [20,20,40,40,40,40]
        self.inrad     = [1.676,3.518,5.423,7.277, 9.081,10.808]
        self.ourad     = [3.518,5.423,7.277,9.081,10.808,12.500]

    # Other known telescopes should be included here, ALMA, ngVLA

        
    def build_panels(self):
        self.rings = []
        for iring in range(self.nrings):
            ring = Ring(self.panelkind,self.npanel[iring],self.inrad[iring],self.ourad[iring])
            ring.create_panels(self.amp,self.dev,self.xaxis,self.yaxis)
            self.rings.append(ring)
        return

        
    def gains(self,wavel):
        # Wavel in cm?
        forpi = 4.0*np.pi
        fact = 1000. * self.reso / wavel
        fact *= fact
        #
        # What are these sums?
        sumrad   = 0.0
        sumtheta = 0.0
        nsamp    = 0.0
        #    convert surface error to phase
        #    and compute gain loss
        for iy in range(self.dev.shape[1]): 
            ycoor = self.yaxis.idx_to_coor(iy)
            for ix in range(self.dev.shape[0]):
                if self.amp[ix,iy]>0:
                    xcoor = self.xaxis.idx_to_coor(ix)
                    rad = np.sqrt(xcoor*xcoor+ycoor*ycoor)
                    if rad > self.diam/2.:
                        continue
                    quo = rad / (2.*self.focus)
                    phase     = self.dev[ix,iy]*forpi /(np.sqrt(1.+quo*quo)*wavel)
                    sumrad   += np.cos(phase)
                    sumtheta += np.sin(phase)
                    nsamp    += 1.0                

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
        rms   = 0.0
        nsamp = 0.0
        for iy in range(self.dev.shape[1]): # what is mx?
            for ix in range(self.dev.shape[0]):
                if self.amp[ix,iy]>0:
                    rms   += self.dev[ix,iy]**2
                    nsamp += 1
                    
        if (nsamp<=0):
            raise Exception("Antenna is blanked")
        rms = np.sqrt(rms/nsamp)
        self.rms = rms
        return rms

    def fit_adjustments(self):
        for iring in range(self.nrings):
            self.rings[iring].solve_panels()
        return 
