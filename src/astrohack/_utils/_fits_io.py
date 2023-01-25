from astropy.io import fits


def _read_fits(filename):
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


def _write_fits(head, data, filename):
    hdu = fits.PrimaryHDU(data)
    hdu.header = head
    hdu.header["ORIGIN"] = "Astrohack"
    hdu.writeto(filename, overwrite=True)
    return
