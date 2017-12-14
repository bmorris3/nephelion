from astroscrappy import detect_cosmics
from astropy.time import Time
from astropy.io import fits
import numpy as np

__all__ = ['RawImage', 'ImageSet']

# Header keywords for the echelle at APO:
date_header_key = 'DATE-OBS'
exposure_header_key = 'EXPTIME'
time_sys_key = 'TIMESYS'
gain_header_key = 'GAIN'
objname_header_key = 'OBJNAME'


class RawImage(object):
    def __init__(self, image=None, header=None, meta=None):
        self.image = image
        self.header = header
        if meta is None:
            meta = dict()
        self.meta = meta

    @property
    def time(self):
        return Time(self.header[date_header_key], format='isot',
                    scale=self.header[time_sys_key].lower())

    @property
    def exposure_duration(self):
        return self.header[exposure_header_key]

    def remove_cosmics(self, sigclip=4.5):
        # http://www.apo.nmsu.edu/arc35m/Instruments/ARCES/
        gain = self.header[gain_header_key]
        readnoise = 2  # e-/pixel
        mask, cleaned_image = detect_cosmics(self.image, gain=gain,
                                             readnoise=readnoise,
                                             sigclip=sigclip,
                                             psfmodel='gaussy',
                                             cleantype='idw')
        return cleaned_image

    @property
    def name(self):
        return self.header[objname_header_key]

    @classmethod
    def from_fits(cls, path):
        image = fits.getdata(path)
        header = fits.getheader(path)
        return cls(image=image, header=header, meta=dict(path=path))


class ImageSet(object):
    def __init__(self, raw_image_list):
        self.raw_images = raw_image_list
        self.stack = None
        self.stack_header = None

    def clean_and_stack(self, *args, **kwargs):
        total_exposure_duration = 0
        image_shape = self.raw_images[0].image.shape
        stacked_image = np.zeros(image_shape)

        times = []
        for image in self.raw_images:
            stacked_image += image.remove_cosmics(*args, **kwargs)
            total_exposure_duration += image.exposure_duration
            times.append(image.time)

        header = self.raw_images[0].header.copy()

        # output times back into the header in TAI:
        header[date_header_key] = Time(Time(times).jd.mean(),
                                       format='jd').tai.isot
        header[exposure_header_key] = total_exposure_duration

        self.stack = stacked_image
        self.stack_header = header
        name = self.raw_images[0].name

        return stacked_image, header, name

