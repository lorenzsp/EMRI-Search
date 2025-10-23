
from few.waveform import GenerateEMRIWaveform
from fastlisaresponse import ResponseWrapper
from lisatools.detector import EqualArmlengthOrbits

default_response_kwargs = dict(
    t0=10000.0,
    order=25,
    tdi="1st generation",
    tdi_chan="AET",
    orbits=EqualArmlengthOrbits(),
)

gen_wave = GenerateEMRIWaveform(
    "FastKerrEccentricEquatorialFlux",
    sum_kwargs=dict(pad_output=True),
)
# parameters
index_lambda = 8
index_beta = 7
T = 1.0
dt = 10.0

response = ResponseWrapper(
    gen_wave,
    T,
    dt,
    index_lambda,
    index_beta,
    flip_hx=True,  # set to True if waveform is h+ - ihx
    remove_sky_coords=False,
    is_ecliptic_latitude=False,
    remove_garbage=True,  # removes the beginning of the signal that has bad information
    **default_response_kwargs,
)