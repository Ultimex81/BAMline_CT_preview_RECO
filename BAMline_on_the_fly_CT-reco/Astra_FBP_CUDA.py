import astra
import matplotlib.pyplot as plt
import numpy

# create geometries and projector
proj_geom = astra.create_proj_geom('parallel', 1.0, 256, numpy.linspace(0, numpy.pi, 1000, endpoint=False))
vol_geom = astra.create_vol_geom(256,256)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

# generate phantom image
#V_exact_id, V_exact = astra.data2d.shepp_logan(vol_geom) tut nicht mit Astra 2.1

# Create a simple hollow cube phantom
cube = numpy.zeros((256,256))
cube[45:213,45:213] = 1
cube[64:197,64:197] = 0
cube[80:180,80:180] = 1
cube[100:160,100:160] = 0

# create forward projection
sinogram_id, sinogram = astra.create_sino(cube, proj_id)

# Display a single projection image
import pylab
pylab.gray()
pylab.figure(1)
pylab.imshow(sinogram)

# reconstruct
recon_id = astra.data2d.create('-vol', vol_geom, 0)
cfg = astra.astra_dict('FBP_CUDA')
cfg['ProjectorId'] = proj_id
cfg['ProjectionDataId'] = sinogram_id
cfg['ReconstructionDataId'] = recon_id
fbp_id = astra.algorithm.create(cfg)
astra.algorithm.run(fbp_id)
V = astra.data2d.get(recon_id)

# Get the result
rec = astra.data2d.get(recon_id)
pylab.figure(2)
pylab.imshow(V)
pylab.show()

# garbage disposal
astra.data2d.delete([sinogram_id, recon_id, V_exact_id])
astra.projector.delete(proj_id)
astra.algorithm.delete(fbp_id)
