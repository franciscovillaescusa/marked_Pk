import numpy as np
import sys,os
import readgadget
import MAS_library as MASL
import smoothing_library as SL
import Pk_library as PKL

################################### INPUT ###############################################
root        = '/simons/scratch/fvillaescusa/pdf_information'
root_out    = '/simons/scratch/fvillaescusa/pdf_information/marked_Pk'
cosmologies = ['fiducial']
ptype       = [1] #CDM
snapnum     = 4 #z=0
grid        = 512
MAS         = 'CIC'
axis        = 0

# smoothing stuff
BoxSize = 1000.0 #Mpc/h
R       = 10.0 #Mpc/h smoothing length
Filter  = 'Top-Hat' #'Top-Hat' or 'Gaussian'
threads = 2
#########################################################################################

# find the redshift of the snapshot
z = {4:0, 3:0.5, 2:1, 1:2, 0:3}[snapnum]

# compute the kernel in Fourier space
W_k = SL.FT_filter(BoxSize, R, grid, Filter, threads)

# do a loop over the different cosmologies
for cosmo in cosmologies:

    # create output folder if it doesnt exists
    if not(os.path.exists('%s/%s'%(root_out,cosmo))):
        os.system('mkdir %s/%s'%(root_out,cosmo))

    # find the number of realizations of each model
    if cosmo=='fiducial':  realizations = 15000
    else:                  realizations = 500

    # do a loop over the different realizations
    for i in xrange(realizations):

        # create output folder if it doesnt exists
        if not(os.path.exists('%s/%s/%d'%(root_out,cosmo,i))): 
            os.system('mkdir %s/%s/%d'%(root_out,cosmo,i))

        # find the snapshot name
        snapshot = '%s/%s/%d/snapdir_%03d/snap_%03d'%(root,cosmo,i,snapnum,snapnum)

        # find name of output file
        fout = '%s/%s/%d/Pk_marked_z=%s.txt'%(root_out,cosmo,i,z)
        if os.path.exists(fout):  continue

        # read header
        header   = readgadget.header(snapshot)
        BoxSize  = header.boxsize/1e3  #Mpc/h

        # read positions, velocities and IDs of the particles
        pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #Mpc/h

        # compute the density field
        delta = np.zeros((grid,grid,grid), dtype=np.float32)
        MASL.MA(pos,delta,BoxSize,MAS)
        delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

        # smoothing the density field (check if numbers are still >-1)
        delta_smoothed = SL.field_smoothing(delta, W_k, threads)
        
        # find the value of the smoothed density field on top of each particle
        weight = np.zeros(pos.shape[0], dtype=np.float32)
        MASL.CIC_interp(delta_smoothed, BoxSize, pos, weight)

        # compute the density field weighing each particle by its weigth
        delta = np.zeros((grid,grid,grid), dtype=np.float32)
        MASL.MA(pos,delta,BoxSize,MAS,W=weight)
        delta /= np.mean(delta,dtype=np.float32);  delta -= 1.0 #not sure about this

        # compute marked Pk
        Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads)
        
        # save results to file
        np.savetxt(fout, np.transpose([Pk.k3D, Pk.Pk[:,0]]))
