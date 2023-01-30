import numpy as numpy


def nullify(matrix, vector, start_par, end_par, npar):
    if start_par == end_par:
        loop = [start_par]
    else:
        loop = [start_par, end_par]
    for ipar in loop:
        vector[ipar] = 0.0
        for jpar in range(npar):
            matrix[jpar, ipar] = 0.0
            matrix[ipar, jpar] = 0.0

    return matrix, vector

def phase_fitting(npix, wavelength, focus, xymin, xymax, cellxy, vamp, vpha, p0, px, py, fx, fy, fz, dp0, dpx, dpy, dfx, dfy, dfz, rms0, rms, ierr, noxy, nofoc, notilt, nopnt, nocass, xmag, xoff, slope, phamod, tx, ty, dtx, dty, cass, dcass):
    #-----------------------------------------------------------------------
    #     FLATPH2 corrects the grading phase for pointing, focus, and feed
    #     offset errors using least squares, and a model incorporating
    #     subreflector position errors.  Includes reference pointing
    #
    #  This is a revised version of the task, offering a two-reflector
    #  solution.  M. Kesteven, 6/12/1994
    #
    #  The formulation is in terms of the Ruze expressions (the unpublished
    #  lecture notes : Small Displacements in Parabolic Antennas, 1969).
    #
    #  At present, this requires the magnification to be HARDWIRED -
    #  see the data statement.
    #
    #     Given:
    #          NPIX        I     Number of pixels on a side of the map.
    #          wavelength      R     Observing wavelength, in meters.
    #          FOCUS       R     Nominal focal length, in meters.
    #          XYMIN       R     Range of |x| and |y| used in correcting for
    #      and XYMAX       R     pointing, focus, and feed offset. Negative
    #                            values denote a range of SQRT(x*x + y*y).
    #          CELLXY      R     Map cell spacing, in meters.
    #          VAMP(N,N)   R     Grading amplitude map.
    #          VPHA(N,N)   R     Grading phase map.
    #          NOPNT       L     Disable phase slope (pointing offset)
    #          NOCASS      l     Disable Cassegrain offsets (X, Y, Z)
    #          NOXY        L     Disable subreflector offset model
    #          NOFOC       L     Disable subreflector focus (z) model
    #          NOTILT      L     Enable subreflector rotation model.
    #          XMAG        R     Magnification (default 13)
    #          XOFF        R     Offset (prime focus to bottom subreflector)
    #          SLOPE       R     Slope to apply to Q
    #     Returned:
    #          P0          R     Constant offset removed, degrees.
    #          PX,PY       R     Least squares estimates of the phase ramp
    #                            in the X and Y directions, in degrees per
    #                            cell.
    #          FX,FY,FZ    R     The derived focal position is at
    #                            (FX,FY,FOCUS+FZ), millimeters.
    #          TX,TY       R     Tilt of subreflector in X, Y axes.
    #          DP0         R     Standard error in P0.
    #          DPX,DPY     R     Standard error in PX, and PY.
    #          DFX,DFY,DFZ R     Standard error in FX, FY, and PZ.
    #          DTX,DTY     R     Standard error in TX, TY.
    #          RMS         R     Weighted Half-path rms error, in mm.
    #          RMS0        R     Pre-fit weighted half-path error, mm.
    #          IERR        I     Error status, 0 means success.
    #          PHMOD(N,N)  R     Model phase, due to subref. offsets
    #          VPHA(N,N)   R     Phase map corrected for subr. offsets.
    #
    #     Called:
    #          APLNOT: {LEASQR}
    #
    #     Algorithm:
    #          Weighted least squares fit.
    #
    #     Notes:
    #       1)  Subreflector offset inhibited if NOXY = .true.
    #       2)  Subreflector focus model inhibited if NOFOC = .true.
    #       3)  Subreflector tilt inhibited if NOTILT = .true.
    #       4)  Phase slope (pointing offset) inhibited if NOPNT = .true.
    #
    #     Author:
    #          Mark Calabretta, Australia Telescope.
    #          Origin; 1987/Nov.    Code last modified; 1989/Nov/01.
    #          mjk, 28/1/93
    #          RAP, 27/05/08
    #-----------------------------------------------------------------------
    # integer   npix, ierr
    # real      wavelength, focus, xymin, xymax, cellxy, vamp(npix,npix),&
    #      & vpha(npix,npix), p0, px, py, fx, fy, fz, dp0, dpx, dpy, dfx,&
    #      & dfy, dfz, rms0, rms, xmag, xoff, slope, phamod(npix,npix),&
    #      & tx, ty, dtx, dty, cass(2), dcass(2)
    # logical   noxy, nofoc, notilt, nopnt, nocass
    # #
    # integer   np
    # parameter (np=10)
    # integer   i, idr2, idx, idy, ix, ix0, ir2max, ir2min, ixymax,&
    #      & ixymin , iy, iy0, j
    # real      corr, fit, fp, m(np,np), ns, ph, r(np), sum, ssq, ssqres,&
    #      & varres, vary, vx(np), wt, x(np), xf, xp, yf, yp, zf, r4,&
    #      & mean, rad, rr, mag, fequiv, ang, q, qp, denom, denomp, xt,&
    #      & yt, xq, xc, yc, cx, cy
    np = 10
    mag = 13.0
    #-----------------------------------------------------------------------
    #   Initialize.
    if xmag <=0.0:
        xmag = mag
    ixymin = abs(xymin/cellxy)
    ixymax = abs(xymax/cellxy)
    ir2min = (xymin*xymin)/(cellxy*cellxy)
    ir2max = (xymax*xymax)/(cellxy*cellxy)
    #   focal length in cellular units
    fp = focus/cellxy
    fequiv = xmag * fp
    #   half-path wavelength scaling
    r4 = wavelength / 720.0
    ns  = 0.0
    sum = 0.0
    ssq = 0.0
    m = numpy.zeros((np,np))
    r = numpy.zeros(np)
    rr = npix/2.
    #   calculate pre-fit rms.
    # call srfrms (npix, xymin, xymax, ixymin, ixymax, ir2min, ir2max,&
    #      & vamp, vpha, r4, mean, rms0)
    # #   loop through the map.
    ix0 = npix/2
    iy0 = npix/2
    for iy in range(npix):
        idy = abs(iy-iy0)
        #   check absolute limits.
        if xymin > 0.0 and idy < ixymin:
            continue
        if xymax > 0.0 and idy > ixymax:
            continue
        #   is this row of pixels outside
        #   the outer ring?
        if xymax < 0.0 and idy*idy > ir2max:
            continue
        for ix in range(npix):
            #   ignore blanked pixels.
            if numpy.isnan(vpha[ix,iy]):
                continue
            #   check for inclusion.
            idx  = abs(ix-ix0)
            idr2 = idx*idx + idy*idy
            #   inner limits.
            if xymin > 0.0:
                if idx < ixymin:
                    continue
            elif xymin < 0.0:
                if idr2 < ir2min:
                    continue
        
            #   outer limits.
            if xymax > 0.0:
                if idx > ixymax:
                    continue
            elif xymax < 0.0:
                if idr2 > ir2max:
                    continue
            #   evaluate variables (in cells)
            ph = vpha[ix, iy]
            wt = vamp[ix, iy]
            xp = ix - ix0
            yp = iy - iy0
            rad = numpy.sqrt(xp*xp + yp*yp)
            ang = numpy.atan2(yp, xp)
            q = rad/(2.*fp)
            qp = q/xmag
            denom = 1.+q*q
            denomp = 1.+qp*qp
            #           xq = (0.3 - 0.7 * q * q) * q
            xq = 0.3 * q
            zf = (1.-q*q)/denom + (1.-qp*qp)/denomp
            #           xf = -2.* numpy.cos(ang) * (xq/denom - qp/denomp)
            #           yf = -2.* numpy.sin(ang) * (xq/denom - qp/denomp)
            xf = -2.* numpy.cos(ang) * (q/denom - slope*q - qp/denomp)
            yf = -2.* numpy.sin(ang) * (q/denom - slope*q - qp/denomp)
            xt = 2.* numpy.cos(ang) * (q/denom + q/denomp)
            yt = 2.* numpy.sin(ang) * (q/denom + q/denomp)
            xc = -2.*numpy.cos(ang)*qp/denomp
            yc = -2.*numpy.sin(ang)*qp/denomp
            #       write(6,100) xp*cellxy,yp*cellxy,rad*cellxy,ang*57.3,q,qp,
            #     1      denom,denomp,zf,xf,yf,xt,yt
            #100    format(1x,3f5.1,f5.0,9f5.2)
            #                                  generate the design matrix.
            ns     = ns  + wt
            sum    = sum + ph*wt
            ssq    = ssq + ph*ph*wt
            r[1]   = r[1] + ph*wt
            r[2]   = r[2] + ph*xp*wt
            r[3]   = r[3] + ph*yp*wt
            r[4]   = r[4] + ph*xf*wt
            r[5]   = r[5] + ph*yf*wt
            r[6]   = r[6] + ph*zf*wt
            r[7]   = r[7] + ph*xt*wt
            r[8]   = r[8] + ph*yt*wt
            r[9]   = r[9] + ph*xc*wt
            r[10]   = r[10] + ph*yc*wt
            m[1,1] = m[1,1] + wt
            m[1,2] = m[1,2] + xp*wt
            m[1,3] = m[1,3] + yp*wt
            m[1,4] = m[1,4] + xf*wt
            m[1,5] = m[1,5] + yf*wt
            m[1,6] = m[1,6] + zf*wt
            m[1,7] = m[1,7] + xt*wt
            m[1,8] = m[1,8] + yt*wt
            m[1,9] = m[1,9] + xc*wt
            m[1,10] = m[1,10] + yc*wt
            m[2,2] = m[2,2] + xp*xp*wt
            m[2,3] = m[2,3] + xp*yp*wt
            m[2,4] = m[2,4] + xp*xf*wt
            m[2,5] = m[2,5] + xp*yf*wt
            m[2,6] = m[2,6] + xp*zf*wt
            m[2,7] = m[2,7] + xp*xt*wt
            m[2,8] = m[2,8] + xp*yt*wt
            m[2,9] = m[2,9] + xp*xc*wt
            m[2,10] = m[2,10] + xp*yc*wt
            m[3,3] = m[3,3] + yp*yp*wt
            m[3,4] = m[3,4] + yp*xf*wt
            m[3,5] = m[3,5] + yp*yf*wt
            m[3,6] = m[3,6] + yp*zf*wt
            m[3,7] = m[3,7] + yp*xt*wt
            m[3,8] = m[3,8] + yp*yt*wt
            m[3,9] = m[3,9] + yp*xc*wt
            m[3,10] = m[3,10] + yp*yc*wt
            m[4,4] = m[4,4] + xf*xf*wt
            m[4,5] = m[4,5] + xf*yf*wt
            m[4,6] = m[4,6] + xf*zf*wt
            m[4,7] = m[4,7] + xf*xt*wt
            m[4,8] = m[4,8] + xf*yt*wt
            m[4,9] = m[4,9] + xf*xc*wt
            m[4,10] = m[4,10] + xf*yc*wt
            m[5,5] = m[5,5] + yf*yf*wt
            m[5,6] = m[5,6] + yf*zf*wt
            m[5,7] = m[5,7] + yf*xt*wt
            m[5,8] = m[5,8] + yf*yt*wt
            m[5,9] = m[5,9] + yf*xc*wt
            m[5,10] = m[5,10] + yf*yc*wt
            m[6,6] = m[6,6] + zf*zf*wt
            m[6,7] = m[6,7] + zf*xt*wt
            m[6,8] = m[6,8] + zf*yt*wt
            m[6,9] = m[6,9] + zf*xc*wt
            m[6,10] = m[6,10] + zf*yc*wt
            m[7,7] = m[7,7] + xt*xt*wt
            m[7,8] = m[7,8] + xt*yt*wt
            m[7,9] = m[7,9] + xt*xc*wt
            m[7,10] = m[7,10] + xt*yc*wt
            m[8,8] = m[8,8] + yt*yt*wt
            m[8,9] = m[8,9] + yt*xc*wt
            m[8,10] = m[8,10] + yt*yc*wt
            m[9,9] = m[9,9] + xc*xc*wt
            m[9,10] = m[9,10] + xc*yc*wt
            m[10,10] = m[10,10] + yc*yc*wt


    #   disable the subreflector
    #   tilt term if requested
    if notilt:
        m, r = nullify(m, r, 6, 7, np)

    #   disable the focus and feed
    #   offset if requested
    if noxy:
        m, r = nullify(m, r, 3, 4, np)

    if nofoc:
        m, r = nullify(m, r, 5, 5, np)

    if nopnt:
        m, r = nullify(m, r, 1, 2, np)

    if nocass:
        m, r = nullify(m, r, 8, 9, np)

    #   compute the least squares
    #   solution.
    call leasqr (np, ns, sum, ssq, r, m, x, vx, ssqres, varres, vary, fit, ierr)
    #
    p0 = x[1]
    px = x[2]
    py = x[3]
    fx = x[4]
    fy = x[5]
    fz = x[6]
    tx = x[7]
    ty = x[8]
    dp0 = numpy.sqrt(vx[1])
    cx = x[9]
    cy = x[10]
    dpx = numpy.sqrt(vx[2])
    dpy = numpy.sqrt(vx[3])
    dfx = wavelength*numpy.sqrt(vx[4])/0.36
    dfy = wavelength*numpy.sqrt(vx[5])/0.36
    dfz = wavelength*numpy.sqrt(vx[6])/0.36
    dtx = wavelength*numpy.sqrt(vx[7])/0.36 * rad2dg / (1000.0 * xoff)
    dty = wavelength*numpy.sqrt(vx[8])/0.36 * rad2dg / (1000.0 * xoff)
    dcass[1] = wavelength*numpy.sqrt(vx[9])/0.36
    dcass[2] = wavelength*numpy.sqrt(vx[10])/0.36
    #   apply the correction.
    for iy in range(npix):
        for ix in range(npix):
            if numpy.isnan(vpha[ix, iy]):
                xp = ix - ix0
                yp = iy - iy0
                rad = numpy.sqrt(xp*xp + yp*yp)
                ang = numpy.atan2(yp, xp)
                q = rad/(2.*fp)
                qp = q/xmag
                denom = 1.+q*q
                denomp = 1.+qp*qp
                xq = (0.3 - 0.7 * q * q) * q
                zf = (1.-q*q)/denom + (1.-qp*qp)/denomp
                xf = -2.* numpy.cos(ang) * (xq/denom - qp/denomp)
                yf = -2.* numpy.sin(ang) * (xq/denom - qp/denomp)
                xt = 2.* numpy.cos(ang) * (q/denom + q/denomp)
                yt = 2.* numpy.sin(ang) * (q/denom + q/denomp)
                xc = -2.*numpy.cos(ang)*qp/denomp
                yc = -2.*numpy.sin(ang)*qp/denomp
                corr = p0 + px*xp + py*yp + fx*xf + fy*yf + fz*zf + tx*xt + ty*yt + cx*xc + cy*yc
                vpha[ix, iy] = vpha[ix,iy] - corr
                phamod[ix, iy] = corr

    #   rescale feed offsets to mm.
    fx = wavelength*fx/0.36
    fy = wavelength*fy/0.36
    fz = wavelength*fz/0.36
    cass[1] = cx * wavelength/0.36
    cass[2] = cy * wavelength/0.36
    #   rescale subr. tilts to degrees
    tx = wavelength*tx/0.36 * rad2dg / (1000.0 * xoff)
    ty = wavelength*ty/0.36 * rad2dg / (1000.0 * xoff)
    #                               rescale phase slope to pointing offset
    px = px/cellxy/360*wavelength*57.296*60.
    py = py/cellxy/360*wavelength*57.296*60.
    #                               compute the post-fit surface rms
    # call srfrms (npix, xymin, xymax, ixymin, ixymax, ir2min, ir2max,&
    #      & vamp, vpha, r4, mean, rms)
    #

#       SUBROUTINE LEASQR (NP, N, SUM, SSQ, R, M, X, VX, SSQRES,
#      *   VARRES, VARY, FIT, IERR)
# C-----------------------------------------------------------------------
# C;  Copyright (C) 1995, 2012
# C;  Associated Universities, Inc. Washington DC, USA.
# C;
# C;  This program is free software; you can redistribute it and/or
# C;  modify it under the terms of the GNU General Public License as
# C;  published by the Free Software Foundation; either version 2 of
# C;  the License, or (at your option) any later version.
# C;
# C;  This program is distributed in the hope that it will be useful,
# C;  but WITHOUT ANY WARRANTY; without even the implied warranty of
# C;  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# C;  GNU General Public License for more details.
# C;
# C;  You should have received a copy of the GNU General Public
# C;  License along with this program; if not, write to the Free
# C;  Software Foundation, Inc., 675 Massachusetts Ave, Cambridge,
# C;  MA 02139, USA.
# C;
# C;  Correspondence concerning AIPS should be addressed as follows:
# C;         Internet email: aipsmail@nrao.edu.
# C;         Postal address: AIPS Project Office
# C;                         National Radio Astronomy Observatory
# C;                         520 Edgemont Road
# C;                         Charlottesville, VA 22903-2475 USA
# C-----------------------------------------------------------------------
# C     LEASQR does the matrix inversion and other necessary tasks
# C     involved in a least squares analysis.
# C
# C     Given:
# C          NP        I     Number of parameters.
# C          N         R     The number of observations.
# C          SUM       R     Error sum.
# C          SSQ       R     Square error sum.
# C          R(NP)     R     Results vector.
# C
# C     Given and returned:
# C          M(NP,NP)  R     On input, the upper triangular part contains
# C                          the design matrix.  This is not changed.
# C                          On output, the lower triangular part contains
# C                          the covariance matrix.  Diagonal elements of
# C                          the covariance matrix are stored in VX.
# C
# C     Returned:
# C          X(NP)     R     Vector holding the least squares solution.
# C          VX(NP)    R     Variance of the best fit parameters.
# C          SSQRES    R     Sum of squares of the residuals.
# C          VARRES    R     Variance of the residuals.
# C          VARY      R     Variance of the error values.
# C          FIT       R     Goodness of fit parameter, lies between 0
# C                          and 1.
# C          IERR      I     Error status, 0 means successful.
# C                             1 - nonspecific error return,
# C                             2 - insufficient degrees of freedom.
# C
# C     Called:
# C          none
# C
# C     Algorithm:
# C          LU-triangular factorization with scaled partial pivoting.
# C          The sub-diagonal triangular matrix contains the scaling
# C          factors used at each step in the Gaussian elimination.  Row
# C          interchanges are recorded in vectors MXS and SXM.
# C             During forward substitution, the pivoting and Gaussian
# C          elimination operations performed on matrix M are applied to
# C          vector R.  Vector X holds the intermediate result.
# C             On backward substitution, successive elements of the
# C          solution vector, X, are calculated by substitution of the
# C          preceding elements into the equations of the upper triangular
# C          factorization of the design matrix.
# C
# C     Notes:
# C       1) Strictly speaking, the design matrix will usually contain
# C          rows of zeroes and therefore be singular.  This arises if no
# C          observations sensitive to a particular parameter have been
# C          done.
# C             In practice, any such singularities are ignored and the
# C          associated parameters remain undetermined.
# C
# C       2) The covariance matrix is the inverse of M(i,j) multiplied by
# C          the variance of the residuals.  It is obtained by forward and
# C          backward substitution on the columns of the unit matrix.
# C
# C       3) Two statement functions, C, and SC have been employed to
# C          partially alleviate the problems posed by passing arrays in
# C          FORTRAN.  The design/covariance matrix m(i,j) is copied into
# C          the working vector s(i).  This is addressed by using C, and
# C          SC in an attempt to make it look like the matrix that it
# C          actually represents.
# C
# C       4) The maximum size problem that LEASQR can handle is set by
# C          parameter MX.= 200.
# C
# C     Author:
# C          Mark Calabretta, Australia Telescope.
# C          Origin; 1988/Sep/29. Code last modified; 1989/Nov/22.
# C-----------------------------------------------------------------------
# C     Parameter which determines the maximum size problem.
#       INTEGER   MX
#       PARAMETER (MX = 200)
# C
#       INTEGER   C, I, IERR, ITEMP, J, K, MXS(MX), NF, NP, PIVOT,
#      *          SXM(MX)
#       REAL      COLMAX, DTEMP, FIT, M(NP,NP), N, R(NP), RESIDU, RLEN,
#      *          ROWMAX(MX), S(MX*MX), SC, RTEMP, SSQ, SSQRES, SUM,
#      *          VARRES, VARY, VX(NP), W(MX), X(NP)
#       INCLUDE 'INCS:DMSG.INC'
# C-----------------------------------------------------------------------
# C     Statement functions for array manipulation, see note 3 above.
#       C(I,J)  = NP*(I-1) + J
#       SC(I,J) = S(C(I,J))
# C-----------------------------------------------------------------------
# C  Initialize.
# C     Anticipate and return immediately on error.
#       IERR = 1
#
# C     Initialize arrays.
#       DO 40 I = 1, NP
# C        Vector which records row interchanges.
#          MXS(I) = I
#
# C        The solution and variance vectors.
#          X(I)  = 0.0
#          VX(I) = 0.0
#
# C        Copy the design matrix and zero the covariance matrix.
#          DO 10 J = 1, I-1
#             M(I,J) = 0.0
#             S(C(I,J)) = M(J,I)
#  10      CONTINUE
#          DO 20 J = I, NP
#             S(C(I,J)) = M(I,J)
#  20      CONTINUE
#
# C        Find the maximum absolute element in each row.
#          ROWMAX(I) = 0.0
#          DO 30 J = 1, NP
#             ROWMAX(I) = MAX(ROWMAX(I), ABS(SC(I,J)))
#  30      CONTINUE
#  40   CONTINUE
#
#       VARY   = 0.0
#       SSQRES = 0.0
#       VARRES = 0.0
#       FIT    = 0.0
#
#
# C     Find the number of degrees of freedom.
#       NF = N
#       DO 60 I = 1, NP
#          IF (ROWMAX(I).NE.0.0) THEN
#             NF = NF - 1
#          ELSE IF (R(I).NE.0.0) THEN
# C           Any row of zeroes must extend to the results vector.
#             WRITE (MSGTXT,50) I
#  50         FORMAT ('LEASQR: Design matrix inconsistency in row',I4)
#             CALL MSGWRT (6)
#          END IF
#  60   CONTINUE
#
#       IF (NF.LE.1) THEN
#          WRITE (MSGTXT,70)
#  70      FORMAT ('LEASQR: Insufficient degrees of freedom.')
#          CALL MSGWRT (6)
#          IERR = 2
#          RETURN
#       END IF
#
#
# C  Factorize the matrix.
#       DO 120 K = 1, NP
# C        Check for a row of zeroes.
#          IF (ROWMAX(K).EQ.0.0) GO TO 120
#
# C        A non-zero row maximum implies non-zero diagonal element.
#          IF (SC(K,K).EQ.0.0) THEN
#             WRITE (MSGTXT,50) MXS(K)
#             CALL MSGWRT (6)
#             GO TO 120
#          END IF
#
# C        Decide whether to pivot.
#          COLMAX = ABS(SC(K,K))/ROWMAX(K)
#          PIVOT = K
#          DO 80 I = K+1, NP
#             IF (ROWMAX(I).NE.0.0) THEN
#                IF (ABS(SC(I,K))/ROWMAX(I).GT.COLMAX) THEN
#                   COLMAX = ABS(SC(I,K))/ROWMAX(I)
#                   PIVOT = I
#                END IF
#             END IF
#  80      CONTINUE
#
#          IF (PIVOT.GT.K) THEN
# C           We must pivot, interchange the rows of the design matrix.
#             DO 90 J = 1, NP
#                DTEMP = SC(PIVOT,J)
#                S(C(PIVOT,J)) = SC(K,J)
#                S(C(K,J)) = DTEMP
#  90         CONTINUE
#
# C           Don't forget the vector of row maxima.
#             DTEMP = ROWMAX(PIVOT)
#             ROWMAX(PIVOT) = ROWMAX(K)
#             ROWMAX(K) = DTEMP
#
# C           Record the interchange for later use.
#             ITEMP = MXS(PIVOT)
#             MXS(PIVOT) = MXS(K)
#             MXS(K) = ITEMP
#          END IF
#
# C        Gaussian elimination.
#          DO 110 I = K+1, NP
# C           Nothing to do if SC(i,k) is zero.
#             IF (SC(I,K).NE.0.0) THEN
# C              Save the scaling factor.
#                S(C(I,K)) = SC(I,K)/SC(K,K)
#
# C              Subtract rows.
#                DO 100 J = K+1, NP
#                   S(C(I,J)) = SC(I,J) - SC(I,K)*SC(K,J)
#  100           CONTINUE
#             END IF
#  110     CONTINUE
#  120  CONTINUE
#
# C     MXS(i) records which row of M corresponds to row i of SC.
# C     SXM(i) records which row of S corresponds to row i of M.
#       DO 130 I = 1, NP
#          SXM(MXS(I)) = I
#  130  CONTINUE
#
#
# C  Solve the normal equations.
#       DO 150 I = 1, NP
# C        Forward substitution.
#          W(I) = R(MXS(I))
#          DO 140 J = 1, I-1
#             W(I) = W(I) - SC(I,J)*W(J)
#  140     CONTINUE
#  150  CONTINUE
#
#       DO 170 I = NP, 1, -1
# C        Backward substitution.
#          IF (SC(I,I).NE.0.0) THEN
#             DO 160 J = I+1, NP
#                W(I) = W(I) - SC(I,J)*W(J)
#  160        CONTINUE
#             W(I) = W(I)/SC(I,I)
#          END IF
#          X(I) = W(I)
#  170  CONTINUE
#
# C     Check that the solution is acceptable.
#       RLEN = 0.0
#       RESIDU = 0.0
#       DO 200 I = 1, NP
#          RTEMP = 0.0
#          DO 180 J = 1, I-1
#             RTEMP = RTEMP + M(J,I)*X(J)
#  180     CONTINUE
#          DO 190 J = I, NP
#             RTEMP = RTEMP + M(I,J)*X(J)
#  190     CONTINUE
#
#          RLEN = RLEN + R(I)**2
#          RESIDU = RESIDU + (RTEMP - R(I))**2
#  200  CONTINUE
#
#       IF (RESIDU.GT.0.001*RLEN) THEN
#          WRITE (MSGTXT,210) RESIDU/RLEN
#  210     FORMAT ('LEASQR: The solution is discrepant at',E8.1)
#          CALL MSGWRT (6)
#          RETURN
#       END IF
#
#
# C  Determine goodness-of-fit estimates, and statistical errors.
#       SSQRES = SSQ
#       DO 220 I = 1, NP
#          SSQRES = SSQRES - X(I)*R(I)
#  220  CONTINUE
#       IF (SSQRES.LT.0.0) SSQRES = 0.0
#
#       VARRES = SSQRES/NF
#       VARY = (SSQ - SUM*SUM/N)/(N - 1.0)
#       FIT = 1.0
#       IF (VARY.NE.0.0) FIT = 1.0 - SSQRES/(SSQ - SUM*SUM/N)
#
# C     Determine the covariance matrix.
#       DO 280 K = 1, NP
# C        Forward substitution affects only that part of W() below the
# C        first non-zero entry.
#          DO 230 I = 1, SXM(K)-1
#             W(I) = 0.0
#  230     CONTINUE
#          W(SXM(K)) = 1.0
#
#          DO 250 I = SXM(K)+1, NP
# C           Forward substitution.
#             W(I) = 0.0
#             DO 240 J = SXM(K), I-1
#                W(I) = W(I) - SC(I,J)*W(J)
#  240        CONTINUE
#  250     CONTINUE
#
#          DO 270 I = NP, K, -1
#             IF (SC(I,I).NE.0.0) THEN
# C              Backward substitution.
#                DO 260 J = I+1, NP
#                   W(I) = W(I) - SC(I,J)*W(J)
#  260           CONTINUE
#                W(I) = W(I)/SC(I,I)
#             END IF
#
#             IF (I.NE.K) THEN
# C              Off diagonal elements of the covariance matrix.
#                M(I,K) = VARRES*W(I)
#             ELSE IF (I.EQ.K) THEN
# C              Diagonal elements of the covariance matrix.
#                VX(K)  = VARRES*W(I)
#             END IF
#  270     CONTINUE
#  280  CONTINUE
#
#
# C  Successful completion.
#       IERR = 0
#
#
#       RETURN
#       END
