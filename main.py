
# Imports
#-------------------------------------------------------------------------------
from __future__ import division
import sys, time, argparse, math, random
import numpy
import numpy as np
from amuse.datamodel import Particles, Particle
from amuse.units import units, nbody_system, constants
from amuse.support.console import set_printing_strategy

from amuse.ic import flatimf, plummer

from amuse.community.ph4.interface import ph4
#from amuse.community.huayno.interface import Huayno as ph4

from amuse.support.codes import stopping_conditions
from amuse.lab import *
from amuse.couple import bridge
import pandas as pd

# Functions
#-------------------------------------------------------------------------------
#path for panda output
path=r'/home/arpan/stellarwind/bondif0.7/'

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("-M", type=float, help="Cluster mass in MSun", default=1.e5)
    parser.add_argument("-N", type=int, help="Number of stars", default=5000)
    parser.add_argument("-R", type=float, help="Cluster radius in pc", default=1.0)
    parser.add_argument("-c_s", type=float, help="Sound speed in gas", default=10.0)

    parser.add_argument("--Mgas", type=float, help="Gas mass in MSun", default=1.e5)
    parser.add_argument("--Rgas", type=float, help="Gas virial radius in pc", default=1.0)

    parser.add_argument("--mdot", type=float, help="Accretion rate in MSun/yr", default=1.e-4)

    parser.add_argument("-t", type=float, help="Simulation time in yr", default=5.e6)
    parser.add_argument("--dt_snapshot", type=float, help="Snapshot interval in yr", default=1.e2)
    parser.add_argument("--dt_bridge", type=float, help="Bridge time step in yr", default=5.e1)

    parser.add_argument("-p", type=int, help="Number of cores", default=1)

    parser.add_argument("-minM", type=float, help="metalicity", default = 10.0)
    parser.add_argument("-maxM", type=float, help="metalicity", default = 100.0)
    parser.add_argument("-Z", type=float, help="metalicity", default = 1.0)

    parser.add_argument("-f", type=str, help="Outfile", default="/home/arpan/stellarwind/bondi0.7/test")



    args = parser.parse_args()
    return args

#-------------------------------------------------------------------------------

def get_radius(m):

    R = 1.6 * (m).value_in(units.MSun)**0.47 | units.RSun

    return R

def get_luminosiy(m):

    #L = 1.03 * (m).value_in(units.MSun)**3.42 | units.LSun
    L = 0.7*3.2e+4*m.value_in(units.MSun) | units.LSun

    return L

def get_temp(m,r):

    L = get_luminosiy(m).value_in(units.W)
    sigma= 5.670374419e-8 #| units.W/(units.m**2.0*units.K**4.0)
    T4 = L/(4.0*math.pi*r.value_in(units.m)**2.0*sigma)
    T2 = math.sqrt(T4)
    T  = math.sqrt(T2) | units.K

    return T


def massloss(m,r):
    L = get_luminosiy(m).value_in(units.LSun)
    T = get_temp(m,r).value_in(units.K)
    massloss=0.0
    if T > 25000.0:
        massloss = -6.697+2.194*math.log10(L/1.e+5)-1.313*math.log10(m.value_in(units.MSun)/30.0)-1.226*math.log10(2.6/2.0)+0.933*math.log10(T/40000.0)-10.92*math.log10(T/40000.0)*math.log10(T/40000.0)+ 0.85*math.log10(Z) 
    else: 
        massloss = -6.688+2.210*math.log10(L/1.e+5)-1.339*math.log10(m.value_in(units.MSun)/30.0)-1.601*math.log10(1.3/2.0)+1.07*math.log10(T/20000.0) + 0.85*math.log10(Z) 

    if m.value_in(units.MSun)>1.0:
        loss= math.pow(10,massloss) | units.MSun/units.yr
    else:
        loss= 1.e-20 | units.MSun/units.yr

    return loss

def get_apocenter(M, R, a, e, l):
    mu = constants.G*M
    ra = (-mu - (mu**2 + 2*e*l**2).sqrt()) / (2*e)
    return ra

def generate_initial_condition(N, Ms, Rs, Mg, Rg, converter, mdot, pot):
    # Generate a Plummer sphere
    stars = plummer.new_plummer_sphere(N, converter)
    reservoir = plummer.new_plummer_sphere(N, converter)

    Rcut = Rs
    As = Rs / 5.
    a2 = As*As

    M = Ms + Mg
    R = Rg

    # Epg=get_energy(stars, background_potential)
    # Eps=stars.potential_energy()
    # Epk=stars.kinetic_energy()
    # Ep=Epg+Eps
    # scale_factor = numpy.sqrt(abs(0.5*Ep) / Epk)
    # stars.velocity *= scale_factor
    # reservoir.velocity *= scale_factor

    Cm = M / stars.mass.sum()
    Cv = math.sqrt(Cm)
    
    for i in range(len(stars)):
        stars[i].vx *= Cv
        stars[i].vy *= Cv
        stars[i].vz *= Cv
        reservoir[i].vx *= Cv
        reservoir[i].vy *= Cv
        reservoir[i].vz *= Cv

    index_remove = []
    for i in range(N):
        orbital_radius = (stars[i].x**2 + stars[i].y**2 + stars[i].z**2).sqrt() 
        orbital_vel = (stars[i].vx**2 + stars[i].vy**2 + stars[i].vz**2).sqrt()
        phi_r = pot.get_potential_at_radius(stars[i].x, stars[i].y, stars[i].z)
        #phi_R = pot.get_potential_at_radius(R, 0 | units.parsec, 0 | units.parsec)
        #vmax  =  (2.*(phi_R-phi_r)).sqrt() 
        #if orbital_vel/vmax >= 0.95:
        #if orbital_radius/R >= 0.99:
        #    index_remove.append(i)                

        lx = stars[i].y*stars[i].vz - stars[i].z*stars[i].vy
        ly = stars[i].z*stars[i].vx - stars[i].x*stars[i].vz
        lz = stars[i].x*stars[i].vy - stars[i].y*stars[i].vx
        l2 = lx**2 + ly**2 + lz**2
        l = l2.sqrt()

        e = 0.5*orbital_vel**2 + phi_r

        #if orbital_radius/As > 2:
            #ra = (-constants.G*Mg - ((constants.G*Mg)**2 + 2*e*l2).sqrt())/(2*e)
            #rp = (-constants.G*Mg + ((constants.G*Mg)**2 + 2*e*l2).sqrt())/(2*e)
        ra = get_apocenter(Mg, Rg, As, e, l)
        #print >> sys.stderr, 'ra =', ra
        if ra/R >= 0.98: 
            index_remove.append(i)                                

    index_remove = sorted(index_remove)

    index_add = []
    if len(index_remove) > 0:
        """
        d = []
        for i in range(N):
            orbital_radius = (reservoir[i].x**2 + reservoir[i].y**2 + reservoir[i].z**2).sqrt() 
            d.append(orbital_radius.value_in(units.parsec))  
        d = sorted(d)
        index_last = 0
        for i in range(N):
            if d[i] >= 0.99*R.value_in(units.parsec):
                index_last = i        
                break

        index_d = []
        for i in range(len(index_remove)):
            index_d.append(index_last-1-i)

        index_add = []
        for j in range(len(index_d)):
            for i in range(N):
                orbital_radius = (reservoir[i].x**2 + reservoir[i].y**2 + reservoir[i].z**2).sqrt() 
                if orbital_radius.value_in(units.parsec) == d[index_d[j]]:
                    index_add.append(i)   
                    break
        """
        index_ra = []
        ras = []    
        for i in range(N):
            orbital_radius = (reservoir[i].x**2 + reservoir[i].y**2 + reservoir[i].z**2).sqrt() 
            orbital_vel = (reservoir[i].vx**2 + reservoir[i].vy**2 + reservoir[i].vz**2).sqrt() 
            phi_r = pot.get_potential_at_radius(reservoir[i].x, reservoir[i].y, reservoir[i].z)

            lx = reservoir[i].y*reservoir[i].vz - reservoir[i].z*reservoir[i].vy
            ly = reservoir[i].z*reservoir[i].vx - reservoir[i].x*reservoir[i].vz
            lz = reservoir[i].x*reservoir[i].vy - reservoir[i].y*reservoir[i].vx
            l2 = lx**2 + ly**2 + lz**2
            l = l2.sqrt()

            e = 0.5*orbital_vel**2 + phi_r

#            if orbital_radius/As > 2:
                #ra = (-constants.G*Mg - ((constants.G*Mg)**2 + 2*e*l2).sqrt())/(2*e)
                #rp = (-constants.G*Mg + ((constants.G*Mg)**2 + 2*e*l2).sqrt())/(2*e)
            ra = get_apocenter(Mg, Rg, As, e, l)

            if ra/R < 0.98:
                ras.append(ra)
                index_ra.append(i)

            #print >> sys.stderr, 'ra =', ra
            #if ra/R < 0.98 and ra/R > 0.65:  ########## Need to fill outer shell ############### 
            #    index_add.append(i)                       
            #    counter += 1
            #if counter == len(index_remove):
            #    break               

        isSorted = False
        while isSorted == False:
            isSorted = True
            for i in range(len(ras)-1):    
                if ras[i] < ras[i+1]:
                    dummy = ras[i]
                    ras[i] = ras[i+1]
                    ras[i+1] = dummy

                    dummy = index_ra[i]
                    index_ra[i] = index_ra[i+1]
                    index_ra[i+1] = dummy

                    isSorted = False

        for i in range(len(index_remove)):
            index_add.append(index_ra[i])

        """
        counter = 0
        for i in range(N):
            orbital_radius = (reservoir[i].x**2 + reservoir[i].y**2 + reservoir[i].z**2).sqrt() 
            #orbital_vel = (reservoir[i].vx**2 + reservoir[i].vy**2 + reservoir[i].vz**2).sqrt() 
            #phi_r = pot.get_potential_at_radius(reservoir[i].x, reservoir[i].y, reservoir[i].z)
            #phi_R = pot.get_potential_at_radius(R, 0 | units.parsec, 0 | units.parsec)
            #vmax  =  (2.*(phi_R-phi_r)).sqrt() 
            #if orbital_vel/vmax < 0.95:
            if orbital_radius/R < 0.99 and orbital_radius/R > 0.7:
                index_add.append(i) 
                counter += 1
            if counter == len(index_remove):
                break               
        index_add = sorted(index_add)
        """
        for i in range(len(index_remove)):
            stars.remove_particle(stars[index_remove[len(index_remove)-1-i]])

        for i in range(len(index_add)):
            stars.add_particle(reservoir[index_add[i]])

    print len(stars)
    stars.move_to_center()

    """
    for s in stars:
        r2 = s.x**2 + s.y**2 + s.z**2
        r = r2.sqrt()

        Menc = M*r*r2 / (r2 + a2)**(3/2.)
        if r > R:
            Menc = M * R**3 / (R**2 + a2)**(3./2)

        myMenc = Menc
        vg2 = constants.G*myMenc / (s.x**2 + s.y**2 + s.z**2).sqrt()
        C = math.sqrt(1. + vg2/(s.vx**2 + s.vy**2 + s.vz**2))
        s.vx *= C
        s.vy *= C
        s.vz *= C
    """
    """
    phi_R = -constants.G*M / (R**2 + a2).sqrt()
    for i in range(N):
        x = stars[i].x
        y = stars[i].y
        z = stars[i].z
        r2 = x**2 + y**2 + z**2

        phi_r = -constants.G*M / (r2 + a2).sqrt()

        vm = (2*(phi_R-phi_r)).sqrt()

        vx = stars[i].vx
        vy = stars[i].vy
        vz = stars[i].vz
        v2 = vx**2 + vy**2 + vz**2
        v  = v2.sqrt()        

        if v/vm > 1.:
            stars[i].vx *= vm/v
            stars[i].vy *= vm/v
            stars[i].vz *= vm/v
    """
    # Set the stellar radii
    for i in range(N):
        stars[i].radius = get_radius(stars[i].mass)

    # Give stars indices
    for i in range(N):
        stars[i].index = str(i)

    # Accretion rate of each star
    for i in range(N):
        stars[i].mdot = bondi_hoyle(stars[i].mass,c_s,stars[i].vx,stars[i].vy,stars[i].vz,Mgas,Rgas,N)

    for i in range(N):
        stars[i].luminosity = get_luminosiy(stars[i].mass)

    for i in range(N):
        stars[i].temperature = get_temp(stars[i].mass, stars[i].radius)

    for i in range(N):
        stars[i].wind = massloss(stars[i].mass, stars[i].radius)


    return stars

#-------------------------------------------------------------------------------

def print_snapshot(t, N, tcpu0, stars, M, R, fo):
    tcpu = time.time()-tcpu0
    header = str(t) + " " + str(N) + " " + str(tcpu) + "\n"
    fo.write(header)
    for s in stars:
        line = s.index + " " + str(s.mass.value_in(units.MSun)) + " " + str(s.mdot.value_in(units.MSun/units.yr)) + " " + str(s.radius.value_in(units.RSun)) + " " + str(s.x.value_in(units.parsec)) + " " + str(s.y.value_in(units.parsec)) + " " + str(s.z.value_in(units.parsec)) + " " + str(s.vx.value_in(units.kms)) + " " + str(s.vy.value_in(units.kms)) + " " + str(s.vz.value_in(units.kms)) + "\n"

        #line = str(s.mass.value_in(units.MSun)) + " " + str(s.radius.value_in(units.RSun)) + " " + str(s.x.value_in(units.parsec)) + " " + str(s.y.value_in(units.parsec)) + " " + str(s.z.value_in(units.parsec)) + " " + str(s.vx.value_in(units.kms)) + " " + str(s.vy.value_in(units.kms)) + " " + str(s.vz.value_in(units.kms)) + "\n"

        fo.write(line)
    fo.write("\n")

#-------------------------------------------------------------------------------

def get_id(stars, key1, key2):
    N = len(stars)
    for i in range(N):
        mykey = stars[i].key
        if mykey == key1:
            index1 = i
        elif mykey == key2:
            index2 = i
    if index2 < index1:
        it = index1
        index1 = index2
        index2 = it
    return index1, index2

# Replace by CoM particle
def handle_encounter(stars, index1, index2, Mgas):
    # Make a CoM particle
    p = Particle()
  
    p.index = stars[index1].index + "_" +  stars[index2].index

    p.mass  = stars[index1].mass + stars[index2].mass

    index_primary = index1
    if stars[index2].mass > stars[index1].mass:
        index_primary = index2

    p.radius = get_radius(p.mass)



    p.wind=massloss(p.mass,p.radius)

    p.luminosity = get_luminosiy(p.mass)

    p.temperature = get_temp(p.mass, p.radius)

    p.x      = (stars[index1].mass*stars[index1].x + stars[index2].mass*stars[index2].x)/p.mass
    p.y      = (stars[index1].mass*stars[index1].y + stars[index2].mass*stars[index2].y)/p.mass
    p.z      = (stars[index1].mass*stars[index1].z + stars[index2].mass*stars[index2].z)/p.mass
    p.vx     = (stars[index1].mass*stars[index1].vx + stars[index2].mass*stars[index2].vx)/p.mass
    p.vy     = (stars[index1].mass*stars[index1].vy + stars[index2].mass*stars[index2].vy)/p.mass
    p.vz     = (stars[index1].mass*stars[index1].vz + stars[index2].mass*stars[index2].vz)/p.mass

    # Remove the collision components
    stars.remove_particle(stars[index2])
    stars.remove_particle(stars[index1])

    if Mgas.value_in(units.MSun)>0.:
        p.mdot = bondi_hoyle(p.mass,c_s,p.vx,p.vy,p.vz,Mgas,Rgas,len(stars))
    else:
        p.mdot = 0. | units.MSun/units.yr

    # Add the collision result
    stars.add_particle(p)

    return stars

#-------------------------------------------------------------------------------

# Potential class
class Potential(object):
    def __init__(self, R = 1 | units.parsec, M = 1.e5 | units.MSun):
        self.M = M
        self.R = R 

        self.a = self.R/5.
        self.a2 = self.a**2

        self.Rv = 16./(3*math.pi)*self.a
        self.Rv2 = self.Rv**2

        self.Rc = self.a / math.sqrt(2.)
        self.Rc2 = self.Rc**2

        self.M0 = M

    def get_gravity_at_point(self, R, x, y, z): 
        r2 = x**2 + y**2 + z**2
        r = r2.sqrt()        

        acc = []
        for i in range(len(r)):
            if r[i] < self.R:
                acc.append( (constants.G * self.M * r[i] / (r2[i] + self.a2)**(3./2)).value_in(units.kms/units.Myr) )
            else:
                Menc = self.M * self.R**3 / (self.R**2 + self.a2)**(3./2)
                acc.append( (constants.G*Menc/r2[i]).value_in(units.kms/units.Myr) )
        acc = acc | units.kms/units.Myr

        ax = -x/r * acc  
        ay = -y/r * acc
        az = -z/r * acc  
        return ax, ay, az

    def get_enclosed_mass_at_point(self, x, y, z):
        r2 = x**2 + y**2 + z**2
        r = r2.sqrt()

        Menc = self.M*r*r2 / (r2 + self.a2)**(3/2.)
        if r > self.R:
            Menc = self.M * self.R**3 / (self.R**2 + self.a2)**(3./2)

        return Menc

    def get_density_at_point(self, x, y, z):
        r2 = x**2 + y**2 + z**2
        r = r2.sqrt()

        rho = 3./(4*math.pi)*self.M0 * self.a2 / (r2 + self.a2)**(5./2)
        if r > self.R:
            rho *= 0.

        return rho
  
    def get_potential_at_radius(self, x, y, z):
        r2 = x**2 + y**2 + z**2
        phi = -constants.G*self.M / (r2 + self.a2).sqrt()
        return phi

    def self_gravity(self):
        phi=-3*math.pi*constants.G*self.M**2.0/(32.0*self.a)
        return phi

def adjust_to_gas(stars, pot):
    """
    Epg=get_energy(stars, background_potential)
    Eps=stars.potential_energy()
    Epk=stars.kinetic_energy()
    Ep=Epg+Eps
    scale_factor = numpy.sqrt(abs(0.5*Ep) / Epk)
    stars.velocity *= scale_factor
    """
    return stars

def get_free_fall_time(M, R):
    V = 4/3.*math.pi*R**3
    rho = 2*M/V # Ms + Mg = 2*M
    tau = ((3*math.pi) / (32*constants.G*rho)).sqrt()
    return tau

def get_time_step_size(args, N, M, Mgas, Rgas, mdot):
    dt1 = args.dt_bridge | units.yr

    sigma2 = constants.G*(Mgas+M)/(2.*Rgas)
    td = 2*Rgas / sigma2.sqrt()
    dt2 = td / 1.e3

    dt3 = get_free_fall_time(Mgas+M, Rgas) / 250.

    dt4 = (Mgas - M) / (N * mdot) / 1.e3

    dt = dt1 # evolve time step
    if dt2 < dt1:
        dt = dt2
    if dt3 < dt:
        dt = dt3
    # if dt4 < dt:
    #     dt = dt4

    dt *= 5.
    dt_snapshot = 10.*dt

    return dt_snapshot, dt, td


def get_energy(stars, background_potential):
    N=len(stars)
    EPg = 0. | units.J
    for i in range(N):
       myphi = stars[i].mass*background_potential.get_potential_at_radius(stars[i].x, stars[i].y, stars[i].z)
       EPg += myphi  

    return EPg


def get_Nstar_still_in_cluster(stars, R):
    R2 = (R)*(R)
    N = 0
    for s in stars:
        r2 = s.x**2 + s.y**2 + s.z**2
        if r2/R2 <= 1.:
            N += 1
    return N

def get_MR_max(stars):
    Mmax = 0. | units.MSun
    Rmax = 0. | units.RSun

    for s in stars:
        if s.mass > Mmax:
            Mmax = s.mass
            Rmax = s.radius

    return Mmax, Rmax

def vel_std(stars):
    vel=[]
    for s in stars:
        v2=s.vx*s.vx+s.vy*s.vy+s.vz*s.vz
        v=v2.sqrt()
        vel.append(v.value_in(units.kms))

    std=np.array(vel)
    sigma= np.std(std) | units.kms

    return sigma


def bondi_hoyle(m,c_s,vx,vy,vz,Mgas,Rgas,N):

    rho=Mgas/(4.0/3.0*math.pi*Rgas**3.0)

    v2=vx*vx+vy*vy+vz*vz

    v=v2.sqrt()

    bh_radius=2.0*constants.G*m/(c_s**2.0+v2)

    seperation=Rgas*math.pow(N,-1.0/3.0)

    #bh=4.0*math.pi*constants.G**2.0*m**2.0*rho/(c_s**2.0+v2)**1.5
    bh= 7.e-9*m.value_in(units.MSun)**2.0*(math.sqrt(c_s.value_in(units.kms)**2.0+v.value_in(units.kms)**2.0)/10.0)**(-3.0) | units.MSun/units.yr

    ratio=seperation.value_in(units.parsec)/bh_radius.value_in(units.parsec)

    alpha=0.
    if ratio>=3.0:
        alpha=0
    elif ratio>=2.0:
        alpha=.18
    elif ratio>=1.0:
        alpha=.3
    elif ratio>0.5:
        alpha=.5 
    elif ratio>0.2:
        alpha=0.6 
    else:
        alpha=1.0

    rate= math.pow(N,alpha)*bh 

    return rate


# Main
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    tcpu0 = time.time()

    # Set output units
    set_printing_strategy("custom", 
                    preferred_units = [units.MSun, units.parsec, units.Myr], 
                    precision = 16, prefix = "", 
                    separator = " [", suffix = "]")

    # Get user options
    args = get_options()
    minM = args.minM | units.MSun
    maxM = args.maxM | units.MSun
    N = args.N                 # Number of stars
    masses=new_salpeter_mass_distribution(N, minM, maxM)
    M = masses.sum()
    #M = args.M | units.MSun    # Total cluster mass
    R = args.R | units.parsec  # Virial radius of cluster
    c_s = args.c_s | units.kms
    Z = args.Z 

    a_star = R/5.
    Rv_star = 16/(3*math.pi)*a_star

    #Mgas = args.Mgas | units.MSun
    Mgas=M
    Rgas = args.Rgas | units.parsec

    a_gas = Rgas/5.
    Rv_gas = 16/(3*math.pi)*a_gas   

    mdot_av = args.mdot | units.MSun/units.yr
    #mdot_c  = mdot_av * 1024. / (60*math.pi)

    mdot = mdot_av
    #mdot0 = mdot_c

    t_end = args.t | units.yr                 # Simulation time
    dt_snapshot = args.dt_snapshot | units.yr # Snapshot time interval

    dt_snapshot, dt, td = get_time_step_size(args, N, M, Mgas, Rgas, mdot_av)
    dt0 = dt
    dt_snapshot0 = dt_snapshot

    print >> sys.stderr, "dt_snap =", dt_snapshot, "dt =", dt

    numCore = args.p

    file_out = args.f
    outfile = file_out + ".run"
    outfile_log = file_out + ".log"
    outfile_colhist = file_out + ".ch"
    outfile_stats = file_out + ".stats"

    # Generate initial condition
    background_potential = Potential(M=Mgas, R=Rgas)
    rho0 = background_potential.get_density_at_point(0. | units.parsec, 0. | units.parsec, 0. | units.parsec)

    converter = nbody_system.nbody_to_si(M, Rv_star)
    stars = generate_initial_condition(N, M, R, Mgas, Rgas, converter, mdot, background_potential)
    stars.mass = masses

    sigma = vel_std(stars) 

    #mdot_sum = 0. | units.MSun/units.yr
    #for s in stars:
    #    myrho = background_potential.get_density_at_point(s.x, s.y, s.z)
    #    mymdot = myrho/rho0*mdot0
    #    mdot_sum += mymdot
    #mdot_sum /= len(stars)

    #print >> sys.stderr, mdot_sum.value_in(units.MSun/units.yr), abs((mdot_sum-mdot)/mdot)
    """
    if ((mdot_sum-mdot)/mdot) > -0.05:
        while ((mdot_sum-mdot)/mdot) > -0.05:
            stars = generate_initial_condition(N, M, R, Mgas, Rgas, converter, mr_mode, mdot)
            mdot_sum = 0. | units.MSun/units.yr
            for s in stars:
                myrho = background_potential.get_density_at_point(s.x, s.y, s.z)
                mymdot = myrho/rho0*mdot0
                mdot_sum += mymdot
            mdot_sum /= len(stars)
            print >> sys.stderr, mdot_sum.value_in(units.MSun/units.yr), abs((mdot_sum-mdot)/mdot)
    """
    numStar0 = len(stars)

    # Setup dynamics 
    grav = ph4(converter, mode='gpu')
    grav.initialize_code()

    grav.particles.add_particles(stars)
    grav.commit_particles()

    # Setup communication channels
    channel_from_grav_to_stars = grav.particles.new_channel_to(stars)
    channel_from_stars_to_grav = stars.new_channel_to(grav.particles)

    # Setup stopping condition for collisions
    sc = grav.stopping_conditions.collision_detection
    sc.enable()

    # Setup background potential
    for i in range(N):
        myrho = background_potential.get_density_at_point(stars[i].x, stars[i].y, stars[i].z)
        stars[i].mdot =  bondi_hoyle(stars[i].mass,c_s,stars[i].vx,stars[i].vy,stars[i].vz,Mgas,Rgas,N)
        stars[i].wind =  massloss(stars[i].mass,stars[i].radius)
        stars[i].radius = get_radius(stars[i].mass)
        stars[i].luminosity = get_luminosiy(stars[i].mass)
        stars[i].temperature = get_temp(stars[i].mass, stars[i].radius)

    # Couple N-body and potential
    integrator = bridge.Bridge(verbose=False)
    integrator.add_system(grav, (background_potential,), True)
    integrator.timestep = dt

    # Adjust stars to gas mass
    stars = adjust_to_gas(stars, background_potential)
    channel_from_stars_to_grav.copy_attributes(['vx', 'vy', 'vz']) 

    # Evolve the system in time
    t_evolve = 0. | units.yr
    t_snapshot = dt_snapshot

    fo = open(outfile, 'w')
    print_snapshot(t_evolve.value_in(units.Myr), len(stars), tcpu0, stars, M, R, fo)

    fo2 = open(outfile_colhist, 'w')
    fo2.write("# t[Myr] ")
    fo2.write("index1 M[MSun] R[RSun] x[pc] y[pc] z[pc] vx[kms] vy[kms] vz[kms] ")
    fo2.write("index2 M[MSun] R[RSun] x[pc] y[pc] z[pc] vx[kms] vy[kms] vz[kms] ")
    fo2.write("index3 M[MSun] R[RSun] x[pc] y[pc] z[pc] vx[kms] vy[kms] vz[kms]\n")

    dEk_enc = 0. | units.J 
    dEp_enc = 0. | units.J 
    dEp_enc_gas= 0. | units.J
    dEk_acc = 0. | units.J
    dEp_acc = 0. | units.J
    dEp_acc_gas= 0. | units.J

    isStellarDominated = False
    t_transition = 0. | units.yr

    isNoGas = False
    t_nogas = 0. | units.yr

    t_last_collision = 0. | units.yr
    Mmax_lastcollision = 0. | units.MSun

    t_evolution = []
    t_evolution.append( t_evolve.value_in(units.Myr) )

    Mstar_evolution = []
    Mstar_evolution.append( stars.mass.sum().value_in(units.MSun) )

    Mgas_evolution = []
    Mgas_evolution.append( background_potential.M.value_in(units.MSun) )

    Rgas_evolution = []
    Rgas_evolution.append( background_potential.R.value_in(units.RSun) )

    mass_max, radius_max = get_MR_max(stars)

    Mmax_evolution = []
    Mmax_evolution.append( mass_max.value_in(units.MSun) )

    Rmax_evolution = []
    Rmax_evolution.append( radius_max.value_in(units.RSun) )

    Ncol_evolution = []
    Ncol_evolution.append( numStar0-len(stars) )

    #Q_evolution = []
    #Q_evolution.append( get_virial_parameter(stars, background_potential) )

    Nenc_evolution = []
    Nenc_evolution.append( get_Nstar_still_in_cluster(stars, 1.0 | units.parsec))

    lr,mf= stars.LagrangianRadii(unit_converter=converter)

    lagrange_radius_50_evolution = []
    lagrange_radius_50_evolution.append( lr[5].value_in(units.parsec) )

    lagrange_radius_10_evolution = []
    lagrange_radius_10_evolution.append(lr[3].value_in(units.parsec))

    lagrange_radius_90_evolution = []
    lagrange_radius_90_evolution.append(lr[7].value_in(units.parsec))

    pos,coreradius,coredens = grav.particles.densitycentre_coreradius_coredens(converter)

    rcore_evolution = []
    rcore_evolution.append(coreradius.value_in(units.parsec))

    density_core_evolution=[]
    density_core_evolution.append(coredens.value_in((units.parsec ** (-3.0))*units.kg))

    potentialstargas_evolution=[]
    potentialstargas_evolution.append(get_energy(stars, background_potential).value_in(units.J))

    kinetic_evolution=[]
    kinetic_evolution.append(grav.kinetic_energy.value_in(units.J))

    potential_evolution=[]
    potential_evolution.append(grav.potential_energy.value_in(units.J))

    selfgravity_evolution=[]
    selfgravity_evolution.append(background_potential.self_gravity().value_in(units.J))

    dEk_coll_evolution=[]
    dEk_coll_evolution.append(dEk_enc.value_in(units.J))

    dEp_coll_evolution=[]
    dEp_coll_evolution.append(dEp_enc.value_in(units.J))

    dEpgas_coll_evolution=[]
    dEpgas_coll_evolution.append(dEp_enc_gas.value_in(units.J))

    dEp_acc_evolution=[]
    dEp_acc_evolution.append(dEp_acc.value_in(units.J))

    dEk_acc_evolution=[]
    dEk_acc_evolution.append(dEk_acc.value_in(units.J))

    dEpgas_acc_evolution=[]
    dEpgas_acc_evolution.append(dEp_acc_gas.value_in(units.J))

    std_evolution = []
    std_evolution.append(sigma.value_in(units.kms))

    dfprop = pd.DataFrame({"mass": [stars.mass.value_in(units.kg)], "radius": [stars.radius.value_in(units.m)], "luminosity": [stars.luminosity.value_in(units.LSun)], "temperature": [stars.temperature.value_in(units.K)], "wind": [stars.wind.value_in(units.MSun/units.yr)], "mdot": [stars.mdot.value_in(units.MSun/units.yr)], "x": [stars.x.value_in(units.m)], "y": [stars.y.value_in(units.m)], "z": [stars.z.value_in(units.m)], "vx": [stars.vx.value_in(units.ms)], "vy": [stars.vy.value_in(units.ms)], "vz": [stars.vz.value_in(units.ms)]} )

    #print >> sys.stderr, "Q=", Q_evolution[0]
    #print >> sys.stderr, "N(<R)=", Nenc_evolution[0]

    Ek_acc = grav.kinetic_energy
    Ep_acc = grav.potential_energy
    Ep_acc_gas = get_energy(stars, background_potential)
    gaspotential = background_potential.self_gravity()


    Nstar_prev = len(stars)
    Nstar_prev_snap = len(stars)
    dt_factor = 1.

    while t_evolve < t_end:
        if len(stars) < numStar0:
            dN = Nstar_prev-len(stars)
            if dN == 1:
                dt_factor /= 2.
            elif dN == 0:
                dt_factor *= 1.5
            if dt_factor*dt >= dt0:
                dt_factor = dt0 / dt
            if dt_factor*dt <= (1./(365.25*24*60) | units.yr):
                dt_factor = (1./(365.25*24*60) | units.yr) / dt
            if dt_factor*dt >= dt_snapshot/10:
                dt_factor = dt_snapshot/10./dt

            dt *= dt_factor
            integrator.timestep = dt

        t_prev = t_evolve
        Nstar_prev = len(stars)

        t_evolve += dt
        #if t_evolve > t_snapshot:
        #    t_evolve = t_snapshot + (1e-5 | units.yr)
        #if t_evolve > t_end:
        #    t_evolve = t_end

        integrator.evolve_model(t_evolve)
        integrator.synchronize_model()
        t_evolve = integrator.model_time
        channel_from_grav_to_stars.copy_attributes(['x', 'y', 'z', 'vx', 'vy', 'vz']) 

        mdot_sum = 0. | units.MSun/units.yr

        Mgas = background_potential.M
        R2cut = Rgas**2

        Ek_acc = grav.kinetic_energy
        Ep_acc = grav.potential_energy
        Ep_acc_gas = get_energy(stars, background_potential)
        gaspotential = background_potential.self_gravity()

        for s in stars:
            r2 = s.x**2 + s.y**2 + s.z**2
            vesc = (2.0*constants.G*s.mass/s.radius).sqrt()
            if r2 < R2cut:
                #myrho = background_potential.get_density_at_point(s.x, s.y, s.z)
                #mymdot = myrho/rho0*mdot0
                s.mdot = bondi_hoyle(s.mass,c_s,s.vx,s.vy,s.vz,Mgas,Rgas,len(stars))
                s.wind = massloss(s.mass,s.radius)

                dm = s.mdot * (t_evolve-t_prev)
                dmloss = s.wind * (t_evolve-t_prev)

                if Mgas.value_in(units.MSun) >= dm.value_in(units.MSun):
                    s.vx= (s.mass*s.vx)/(s.mass+dm-dmloss)
                    s.vy= (s.mass*s.vy)/(s.mass+dm-dmloss)
                    s.vz= (s.mass*s.vz)/(s.mass+dm-dmloss)

                    s.mass += dm-dmloss
                    #if (t_evolve-t_prev).value_in(units.Myr) > 0:
                    #    s.mdot = dm/(t_evolve-t_prev)
                    s.radius = get_radius(s.mass)
                    Mgas -= dm

                    mdot_sum += dm / (t_evolve-t_prev)
                elif Mgas.value_in(units.MSun) > 0.:
                    dm = Mgas
                    s.vx= (s.mass*s.vx)/(s.mass+dm-dmloss)
                    s.vy= (s.mass*s.vy)/(s.mass+dm-dmloss)
                    s.vz= (s.mass*s.vz)/(s.mass+dm-dmloss)
                    s.mass += dm-dmloss
                    s.radius = get_radius(s.mass)
                    Mgas -= dm

                    mdot_sum += dm / (t_evolve-t_prev)
                else:
                    s.mdot = 0. | units.MSun/units.yr
                    s.vx= (s.mass*s.vx)/(s.mass-dmloss)
                    s.vy= (s.mass*s.vy)/(s.mass-dmloss)
                    s.vz= (s.mass*s.vz)/(s.mass-dmloss)

                    s.mass -= dmloss
                    s.radius = get_radius(s.mass)

            else:
                s.mdot = 0. | units.MSun/units.yr
                s.wind = massloss(s.mass,s.radius)
                dmloss = s.wind * (t_evolve-t_prev)

                s.vx= (s.mass*s.vx)/(s.mass-dmloss)
                s.vy= (s.mass*s.vy)/(s.mass-dmloss)
                s.vz= (s.mass*s.vz)/(s.mass-dmloss)

                s.mass -= dmloss
                s.radius = get_radius(s.mass)              

        channel_from_stars_to_grav.copy_attributes(['mass', 'radius', 'vx', 'vy', 'vz']) 


        background_potential.M = Mgas

        # totalchange=(Ek_acc - grav.kinetic_energy)+(Ep_acc - grav.potential_energy) 

        # gaschange=0.5*gaspotential+totalchange

        # if Mgas.value_in(units.MSun)>0.:
        #     dummy= -15.0*math.pi*constants.G*Mgas**2.0/(64.0*gaschange)
        #     if dummy.value_in(units.parsec)<.05:
        #         background_potential.R =.05 | units.parsec
        #     else:
        #         background_potential.R= dummy

        # else:
        #     background_potential.R=0.05 | units.parsec

        if Mgas.value_in(units.MSun)> 0.:
            dEk_acc += (Ek_acc - grav.kinetic_energy)
            dEp_acc += (Ep_acc - grav.potential_energy)
            dEp_acc_gas += (Ep_acc_gas - get_energy(stars, background_potential))
        else:
            dEk_acc += 0.*Ek_acc
            dEp_acc += 0.*Ek_acc
            dEp_acc_gas += 0.*Ek_acc

        #print >> sys.stderr, mdot_sum/1e6/len(stars)

        # Detect and handle collisions
        if sc.is_set():
            collision_data = []
            Ek_enc = grav.kinetic_energy
            Ep_enc = grav.potential_energy
            Ep_enc_gas= get_energy(stars, background_potential)

            # Detect which stars collided
            key1 = sc.particles(0).key
            key2 = sc.particles(1).key
            index1, index2 = get_id(stars, key1, key2)  

            collision_data.append(t_evolve.value_in(units.Myr))

            collision_data.append(index1)
            collision_data.append(stars[index1].mass.value_in(units.MSun))
            collision_data.append(stars[index1].radius.value_in(units.RSun))
            collision_data.append(stars[index1].x.value_in(units.parsec))
            collision_data.append(stars[index1].y.value_in(units.parsec))
            collision_data.append(stars[index1].z.value_in(units.parsec))
            collision_data.append(stars[index1].vx.value_in(units.kms))
            collision_data.append(stars[index1].vy.value_in(units.kms))
            collision_data.append(stars[index1].vz.value_in(units.kms))

            collision_data.append(index2)
            collision_data.append(stars[index2].mass.value_in(units.MSun))
            collision_data.append(stars[index2].radius.value_in(units.RSun))
            collision_data.append(stars[index2].x.value_in(units.parsec))
            collision_data.append(stars[index2].y.value_in(units.parsec))
            collision_data.append(stars[index2].z.value_in(units.parsec))
            collision_data.append(stars[index2].vx.value_in(units.kms))
            collision_data.append(stars[index2].vy.value_in(units.kms))
            collision_data.append(stars[index2].vz.value_in(units.kms))

            # Process the collision in stars
            stars = handle_encounter(stars, index1, index2, Mgas)

            # Update the particle set in grav
            grav.particles.remove_particle(grav.particles[index2])
            grav.particles.remove_particle(grav.particles[index1])
            grav.particles.add_particle(stars[len(stars)-1])

            grav.recommit_particles()

            dEk_enc += (Ek_enc - grav.kinetic_energy) 
            dEp_enc += (Ep_enc - grav.potential_energy)
            dEp_enc_gas += (Ep_enc_gas - get_energy(stars, background_potential))


            index3 = len(stars)-1
            collision_data.append(index3)
            collision_data.append(stars[index3].mass.value_in(units.MSun))
            collision_data.append(stars[index3].radius.value_in(units.RSun))
            collision_data.append(stars[index3].x.value_in(units.parsec))
            collision_data.append(stars[index3].y.value_in(units.parsec))
            collision_data.append(stars[index3].z.value_in(units.parsec))
            collision_data.append(stars[index3].vx.value_in(units.kms))
            collision_data.append(stars[index3].vy.value_in(units.kms))
            collision_data.append(stars[index3].vz.value_in(units.kms)) 

            for cd in collision_data:
                fo2.write(str(cd) + " ")
            fo2.write("\n")

            t_last_collision = t_evolve
            Mmax_lastcollision = stars.mass.max().value_in(units.MSun)

            #print >> sys.stderr, t_evolve, '/', t_end, len(stars), "Collision:", index1, index2

        if t_evolve >= t_snapshot:
            print_snapshot(t_evolve.value_in(units.Myr), len(stars), tcpu0, stars, M, R, fo)

            t_evolution.append( t_evolve.value_in(units.Myr) )
            Mstar_evolution.append( stars.mass.sum().value_in(units.MSun) )
            Mgas_evolution.append( background_potential.M.value_in(units.MSun) )
            Rgas_evolution.append( background_potential.R.value_in(units.RSun) )
            mass_max, radius_max = get_MR_max(stars)
            Mmax_evolution.append( mass_max.value_in(units.MSun) )
            Rmax_evolution.append( radius_max.value_in(units.RSun) )
            Ncol_evolution.append( numStar0-len(stars) )
            Nenc_evolution.append( get_Nstar_still_in_cluster(stars, 1.0 | units.parsec))
            #Q_evolution.append( get_virial_parameter(stars, background_potential) )
            lr,mf= stars.LagrangianRadii(unit_converter=converter)
            lagrange_radius_50_evolution.append( lr[5].value_in(units.parsec) )
            lagrange_radius_10_evolution.append(lr[3].value_in(units.parsec))
            lagrange_radius_90_evolution.append(lr[7].value_in(units.parsec))
            pos,coreradius,coredens = grav.particles.densitycentre_coreradius_coredens(converter)
            rcore_evolution.append(coreradius.value_in(units.parsec))
            density_core_evolution.append(coredens.value_in((units.parsec **(-3.0))*units.kg))
            potentialstargas_evolution.append(get_energy(stars, background_potential).value_in(units.J))
            kinetic_evolution.append(grav.kinetic_energy.value_in(units.J))
            potential_evolution.append(grav.potential_energy.value_in(units.J))
            selfgravity_evolution.append(background_potential.self_gravity().value_in(units.J))

            dEk_acc_evolution.append(dEk_acc.value_in(units.J))
            dEp_acc_evolution.append(dEp_acc.value_in(units.J))
            dEpgas_acc_evolution.append(dEp_acc_gas.value_in(units.J))

            dEk_coll_evolution.append(dEk_enc.value_in(units.J))
            dEp_coll_evolution.append(dEp_enc.value_in(units.J))
            dEpgas_coll_evolution.append(dEp_enc_gas.value_in(units.J))


            sigma=vel_std(stars)
            std_evolution.append(sigma.value_in(units.kms))

            dfprop = dfprop.append({"mass": stars.mass.value_in(units.kg), "radius": stars.radius.value_in(units.m), "luminosity": stars.luminosity.value_in(units.LSun), "temperature": stars.temperature.value_in(units.K), "wind": stars.wind.value_in(units.MSun/units.yr), "mdot": stars.mdot.value_in(units.MSun/units.yr), "x": stars.x.value_in(units.m), "y": stars.y.value_in(units.m), "z": stars.z.value_in(units.m), "vx": stars.vx.value_in(units.ms), "vy": stars.vy.value_in(units.ms), "vz": stars.vz.value_in(units.ms)},ignore_index=True )
              
            if len(stars) < numStar0:
                dQ = Nstar_prev_snap-len(stars)
                if dQ > 5:
                    dt_snapshot /= 4.
                elif dQ < 2:
                    dt_snapshot *= 2.
                else:
                    dt_snapshot *= 1.
                if dt_snapshot >= dt_snapshot0:
                    dt_snapshot = dt_snapshot0

            t_snapshot += dt_snapshot

            Nstar_prev_snap = len(stars)

            #print >> sys.stderr, (stars[0].x**2 + stars[0].y**2 + stars[0].z**2).sqrt()
            if numStar0 > len(stars) and t_evolve/t_last_collision > 1.:
                print >> sys.stderr, t_evolve, '/', t_end, len(stars), stars.mass.max().value_in(units.MSun), stars.mass.min().value_in(units.MSun), stars.radius.max().value_in(units.RSun), stars.mass.sum().value_in(units.MSun), background_potential.M.value_in(units.MSun), mdot.value_in(units.MSun/units.yr), stars.wind.max().value_in(units.MSun/units.yr), background_potential.R.value_in(units.parsec), (1./(t_evolve-t_last_collision))/((numStar0-len(stars))/t_last_collision)
            else:
                print >> sys.stderr, t_evolve, '/', t_end, len(stars), stars.mass.max().value_in(units.MSun), stars.mass.min().value_in(units.MSun), stars.radius.max().value_in(units.RSun), stars.mass.sum().value_in(units.MSun), background_potential.M.value_in(units.MSun), mdot.value_in(units.MSun/units.yr), stars.wind.max().value_in(units.MSun/units.yr), background_potential.R.value_in(units.parsec)

        if len(stars) < numStar0 and t_evolve/t_last_collision > 1.:
            R = 1./(t_evolve-t_last_collision)
            R_av = (numStar0-len(stars))/t_last_collision
            if R/R_av < 0.015:
                t_end = t_last_collision
                break

        """
        if len(stars) < 10:
            break

        if get_Nstar_still_in_cluster(stars, 2*Rgas) < 10:
            if t_evolve-t_last_collision >= 5*td or t_evolve-t_last_collision > 0.1 | units.Myr:
                t_end = t_last_collision
                print >> sys.stderr, t_end.value_in(units.Myr)
                break              

        if isStellarDominated == False:
            Ms_tot = stars.mass.sum().value_in(units.MSun)
            Mg_tot = background_potential.M.value_in(units.MSun) 
            if Ms_tot >= Mg_tot:
                isStellarDominated = True
                t_transition = t_evolve
                #tacc = t_evolve
                #t_end = t_evolve + tacc + 20*td
        if isStellarDominated == True and t_evolve > 2*t_transition:
            if t_evolve-t_last_collision >= 5*td or t_evolve-t_last_collision > 0.1 | units.Myr:
                t_end = t_last_collision
                print >> sys.stderr, t_end.value_in(units.Myr)
                break  
        """

        #if isStellarDominated == True and isNoGas == False:
        #    if t_evolve >= 2*t_transition:
        #        isNoGas = True
        #        t_nogas = t_evolve

        #        ek = stars.kinetic_energy()
        #        M  = stars.mass.sum()
        #        V2 = 2*ek/M
        #        V  = V2.sqrt()
        #        td = 2*Rgas/V

        #        t_end = t_evolve + 5*td

        #print >> sys.stderr, t_evolve, '/', t_end, len(stars)

        #Mstar = 0. | units.MSun
        #for s in stars:
        #    r2 = s.x**2 + s.y**2 + s.z**2
        #    r = r2.sqrt()
        #    if r/Rgas < 1.:
        #        Mstar += s.mass
        #dt = get_time_step_size(args, N, Mstar, Mgas, Rgas)

    # Cleanup
    integrator.stop()
    dfprop.to_csv(path + "properties.csv", index=False)
    dfprop.to_pickle(path + "properties.csv")

    fo.close()
    fo2.close()

    fo4 = open(outfile_log, 'w')

    tcpu1 = time.time()
    fo4.write("N = "+ str(args.N) + "\n")
    fo4.write("M = "+ str(args.M) + "\n")
    fo4.write("R = "+ str(args.R) + "\n")

    fo4.write("Mgas = "+ str(args.Mgas) + "\n")
    fo4.write("Rgas = "+ str(args.Rgas) + "\n")
    fo4.write("mdot = "+ str(args.mdot) + "\n")

    fo4.write("t = "+ str(args.t) + "\n")
    fo4.write("dt_snapshot = "+ str(args.dt_snapshot) + "\n")
    fo4.write("dt_bridge = "+ str(args.dt_bridge) + "\n")

    fo4.write("f = "+ args.f + "\n")
    fo4.write("p = "+ str(args.p) + "\n")

    fo4.write("tcpu = " + str(tcpu1-tcpu0) + "[s]\n")

    numStar = len(stars)
    numCol = numStar0 - numStar
    fo4.write("Ncol = " + str(numCol) + "\n")

    Mmax = 0.
    for s in stars:
        if s.mass.value_in(units.MSun) > Mmax:
            Mmax = s.mass.value_in(units.MSun)
    fo4.write("Mmax = " + str(Mmax_lastcollision) + "\n")

    Rmax = 0.
    for s in stars:
        if s.radius.value_in(units.RSun) > Rmax:
            Rmax = s.radius.value_in(units.RSun)
    fo4.write("Rmax = " + str(Rmax) + "\n")

    fo4.write("Tcross = " + str(td) + "\n")
    fo4.write("Ttransition = " + str(t_transition) + "\n")
    fo4.write("Tlastcollision = " + str(t_end) + "\n")

    fo4.close()

    dfout = pd.DataFrame({"t[Myr]": t_evolution, "M_star[MSun]": Mstar_evolution, "M_gas[MSun]": Mgas_evolution, "R_gas[RSun]": Rgas_evolution, "M_max[MSun]": Mmax_evolution, "R_of_max[RSun]": Rmax_evolution , "Ncol": Ncol_evolution, "Nenc": Nenc_evolution , "lagrange10": lagrange_radius_10_evolution,  "lagrange50": lagrange_radius_50_evolution,  "lagrange90": lagrange_radius_90_evolution, "radiuscore": rcore_evolution, "densitycore":density_core_evolution, "potential_star_gas":potentialstargas_evolution, "kineticstar":kinetic_evolution, "potentialstar":potential_evolution, "selfgravity":selfgravity_evolution, "dEkcoll":dEk_coll_evolution, "dEpcoll":dEp_coll_evolution, "dEgascoll":dEpgas_coll_evolution, "dEkacc":dEk_acc_evolution, "dEpacc":dEp_acc_evolution, "dEgasacc":dEpgas_acc_evolution, "std": std_evolution})
    dfout.to_csv(path + "output.csv", index=False)
    dfout.to_pickle(path + "output.csv")
