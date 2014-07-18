""" 2D Kelvin Helmoltz Instability example """
import numpy

# Particle generator
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline, WendlandQuintic, Gaussian

# SPH Equations and Group
from pysph.sph.equation import Group

from pysph.sph.wc.viscosity import ClearyArtificialViscosity

from pysph.sph.wc.transport_velocity import SummationDensity, MomentumEquationPressureGradient,\
    SolidWallPressureBC, SolidWallNoSlipBC, ShepardFilteredVelocity, \
    StateEquation, MomentumEquationArtificialStress

from pysph.sph.surface_tension import ColorGradientUsingNumberDensity, \
    InterfaceCurvatureFromNumberDensity, ShadlooYildizSurfaceTensionForce, \
    DiscretizedDiracDelta

from pysph.sph.gas_dynamics.basic import ScaleSmoothingLength

# PySPH solver and application
from pysph.solver.application import Application
from pysph.solver.solver import Solver

# Integrators and Steppers
from pysph.sph.integrator_step import VerletSymplecticWCSPHStep, TransportVelocityStep
from pysph.sph.integrator import PECIntegrator

# Domain manager for periodic domains
from pysph.base.nnps import DomainManager

# problem parameters
dim = 2
domain_width = 1.0
domain_height = 1.0

# numerical constants
alpha = 0.001
wavelength = 1.0
wavenumber = 2*numpy.pi/wavelength
Ri = 0.1
rho0 = rho1 = 1000.0
rho2 = rho1
U = 0.5
sigma = Ri * (rho1*rho2) * (2*U)**2/(wavenumber*(rho1 + rho2))
psi0 = 0.03*domain_height

# discretization parameters
nghost_layers = 5
dx = dy = 0.01
dxb2 = dyb2 = 0.5 * dx
volume = dx*dx
hdx = 1.3
h0 = hdx * dx
rho0 = 1000.0
c0 = 25.0
p0 = c0*c0*rho0
nu = 0.125 * alpha * h0 * c0

# time steps
tf = 2.0
dt_cfl = 0.25 * h0/( 1.1*c0 )
dt_viscous = 0.125 * h0**2/nu
dt_force = 1.0

dt = 0.9 * min(dt_cfl, dt_viscous, dt_force)

def create_particles(**kwargs):
    ghost_extent = (nghost_layers + 0.5)*dx

    x, y = numpy.mgrid[ dxb2:domain_width:dx, -ghost_extent:domain_height+ghost_extent:dy ]
    x = x.ravel(); y = y.ravel()

    m = numpy.ones_like(x) * volume * rho0
    rho = numpy.ones_like(x) * rho0
    h = numpy.ones_like(x) * h0
    cs = numpy.ones_like(x) * c0

    # additional properties required for the fluid.
    additional_props = [
        # volume inverse or number density
        'V', 

        # color and gradients
        'color', 'cx', 'cy', 'cz', 'cx2', 'cy2', 'cz2',
        
        # discretized interface normals and dirac delta
        'nx', 'ny', 'nz', 'ddelta',

        # interface curvature
        'kappa',
        
        # filtered velocities
        'uf', 'vf', 'wf', 
        
        # transport velocities
        'uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat', 
        
        # imposed accelerations on the solid wall
        'ax', 'ay', 'az', 'wij', 
       
        # velocity of magnitude squared
        'vmag2',
        
        ]

    # get the fluid particle array
    fluid = get_particle_array(
        name='fluid', x=x, y=y, h=h, m=m, rho=rho, cs=cs, 
        additional_props=additional_props)

    # set the fluid velocity with respect to the sinusoidal
    # perturbation
    fluid.u[:] = -U
    mode = 1
    for i in range(len(fluid.x)):
        if fluid.y[i] > domain_height/2 + psi0*domain_height*numpy.sin(2*numpy.pi*fluid.x[i]/(mode*domain_width)):
            fluid.u[i] = U
            fluid.color[i] = 1

    # extract the top and bottom boundary particles
    indices = numpy.where( fluid.y > domain_height )[0]
    wall = fluid.extract_particles( indices )
    fluid.remove_particles( indices )

    indices = numpy.where( fluid.y < 0 )[0]
    bottom = fluid.extract_particles( indices )
    fluid.remove_particles( indices )
    
    # concatenate the two boundaries
    wall.append_parray( bottom )
    wall.set_name( 'wall' )

    # set the number density initially for all particles
    fluid.V[:] = 1./volume
    wall.V[:] = 1./volume

    # set additional output arrays for the fluid
    fluid.add_output_arrays(['V', 'color', 'cx', 'cy', 'nx', 'ny', 'ddelta'])
    
    print "2D KHI with %d fluid particles and %d wall particles"%(
            fluid.get_number_of_particles(), wall.get_number_of_particles())

    return [fluid, wall]
    
# domain for periodicity
domain = DomainManager(xmin=0, xmax=domain_width, ymin=0, ymax=domain_height,
                       periodic_in_x=True, periodic_in_y=False)

# Create the application.
app = Application(domain=domain)

# Create the kernel
kernel = WendlandQuintic(dim=2)

# Create the Integrator.
integrator = PECIntegrator( fluid=TransportVelocityStep() )

# Create a solver.
solver = Solver(
    kernel=kernel, dim=dim, integrator=integrator,
    dt=dt, tf=tf, adaptive_timestep=False)

# create the equations
equations = [

    # We first compute the mass and number density of the fluid
    # phase. This is used in all force computations henceforth. The
    # number density (1/volume) is explicitly set for the solid phase
    # and this isn't modified for the simulation.
    Group(equations=[
            SummationDensity( dest='fluid', sources=['fluid', 'wall'] )
            ] ),
    
    # Given the updated number density for the fluid, we can update
    # the fluid pressure. Additionally, we can compute the Shepard
    # Filtered velocity required for the no-penetration boundary
    # condition. Additionally, we compute the gradient of the color
    # function with respect to the original smoothing length. This
    # will compute the interface normals.
    Group(equations=[
            StateEquation(dest='fluid', sources=None, rho0=rho0, p0=p0),
            ShepardFilteredVelocity(dest='fluid', sources=['fluid']),
            ColorGradientUsingNumberDensity(dest='fluid', sources=['fluid', 'wall']),
            ] ),

    #################################################################
    # Begin Surface tension formulation
    #################################################################
    # Scale the smoothing lengths to determine the interface
    # quantities. The NNPS is updated after this group to get the new
    # list of neighbors.
    Group(equations=[
            ScaleSmoothingLength(dest='fluid', sources=None, factor=0.8)
            ], update_nnps=True ),

    # Compute the discretized dirac delta function and the interface
    # curvature using the modified smoothing length.
    Group(equations=[
            DiscretizedDiracDelta(dest='fluid', sources=['fluid', 'wall']),
            InterfaceCurvatureFromNumberDensity(dest='fluid', sources=['fluid']),
            ], ),

    # Now rescale the smoothing length to the original
    # value. Re-compute NNPS at the end of this Group since the
    # smoothing lengths are modified.
    Group(equations=[
            ScaleSmoothingLength(dest='fluid', sources=None, factor=1.25)
            ], update_nnps=True,
          ),
    #################################################################
    # End Surface tension formulation
    #################################################################

    # Once the pressure for the fluid phase has been updated, we can
    # extrapolate the pressure to the wall ghost particles. After this
    # group, the density and pressure of the boundary particles has
    # been updated and can be used in the integration equations.
    Group(
        equations=[
            SolidWallPressureBC(dest='wall', sources=['fluid'], p0=p0, rho0=rho0),
            
            ], ),
    
    # The main acceleration block
    Group(
        equations=[

            # Gradient of pressure for the fluid phase using the
            # number density formulation. No penetration boundary
            # condition using Adami et al's generalized wall boundary
            # condition. The extrapolated pressure and density on the
            # wall particles is used in the gradient of pressure to
            # simulate a repulsive force.
            MomentumEquationPressureGradient(
                dest='fluid', sources=['fluid', 'wall'], pb=p0),

            # Artificial viscosity for the fluid phase.
            ClearyArtificialViscosity(dest='fluid', sources=['fluid'], dim=dim, alpha=alpha),

            # No-slip boundary condition using Adami et al's
            # generalized wall boundary condition. This equation
            # basically computes the viscous contribution on the fluid
            # from the wall particles.
            SolidWallNoSlipBC(dest='fluid', sources=['wall'], nu=nu),
                                      
            # Surface tension force for the SY11 formulation
            ShadlooYildizSurfaceTensionForce(dest='fluid', sources=None, sigma=sigma),

            # Artificial stress for the fluid phase
            MomentumEquationArtificialStress(dest='fluid', sources=['fluid']),
                                               
            ], )
    ]

# Setup the application and solver.  This also generates the particles.
app.setup(solver=solver, equations=equations,
          particle_factory=create_particles)

app.run()
