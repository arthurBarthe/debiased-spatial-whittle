import autograd.numpy as np
import matplotlib.pyplot as plt
from debiased_spatial_whittle.grids import RectangularGrid
from debiased_spatial_whittle.models import ExponentialModel, SquaredExponentialModel, MaternModel, MaternCovarianceModel
from debiased_spatial_whittle.plotting_funcs import plot_marginals
from debiased_spatial_whittle.bayes import DeWhittle, Whittle, Gaussian
from debiased_spatial_whittle.simulation import SamplerOnRectangularGrid

np.random.seed(152341)



# plt.figure()
# plt.imshow(z, cmap='Spectral')
# plt.show()

params = np.array([10.,1.])

def dewhittle_test():
    model = SquaredExponentialModel()
    model.rho = 10
    model.sigma = 1
    model.nugget=0.1
    grid = RectangularGrid((64,64))
    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()
    
    dw = DeWhittle(z, grid, SquaredExponentialModel(), nugget=0.1)
    dw.fit(None, prior=False)
    MLEs = dw.sim_MLEs(params, niter=50)
    plot_marginals([MLEs], np.log(params))

def dewhittle_matern_conditional_nu_test():
    sim_model = MaternModel()
    sim_model.rho = 10
    sim_model.sigma = 1
    sim_model.nu = 3/2
    sim_model.nugget=0.1
    grid = RectangularGrid((64,64))
    sampler = SamplerOnRectangularGrid(sim_model, grid)
    z = sampler()
    
    model = MaternModel()
    model.nu =3/2
    dw = DeWhittle(z, grid, model, nugget=0.1)
    dw.fit(None, prior=False)
    MLEs = dw.sim_MLEs(params, niter=50)
    plot_marginals([MLEs], np.log(params))
    
def dewhittle_matern_test():
    sim_model = MaternModel()
    sim_model.rho = 10
    sim_model.sigma = 1
    sim_model.nu = 3/2
    sim_model.nugget=0.1
    grid = RectangularGrid((64,64))
    sampler = SamplerOnRectangularGrid(sim_model, grid)
    z = sampler()
    
    params = np.array([10.,1.,3/2])
    model = MaternModel()
    # model.nu =3/2
    dw = DeWhittle(z, grid, model, nugget=0.1)
    dw.fit(np.log(params), prior=False, approx_grad=True)
    MLEs = dw.sim_MLEs(params, niter=50, approx_grad=True)
    plot_marginals([MLEs], np.log(params))
    

def whittle_test():
    model = SquaredExponentialModel()
    model.rho = 10
    model.sigma = 1
    model.nugget=0.1
    grid = RectangularGrid((64,64))
    sampler = SamplerOnRectangularGrid(model, grid)
    z = sampler()
    
    whittle = Whittle(z, grid, SquaredExponentialModel(), nugget=0.1)
    whittle.fit(None, prior=False)
    # stop
    MLEs = whittle.sim_MLEs(params, niter=50)
    plot_marginals([MLEs], np.log(params))
    
    
def whittle_matern_test():    # TODO: wrong spectral density
    sim_model = MaternModel()
    sim_model.rho = 10
    sim_model.sigma = 1
    sim_model.nu = 3/2
    sim_model.nugget=0.1
    grid = RectangularGrid((64,64))
    sampler = SamplerOnRectangularGrid(sim_model, grid)
    z = sampler()
    
    params = np.array([10.,1.,3/2])
    model = MaternModel()
    # model.nu =3/2
    whittle = Whittle(z, grid, model, nugget=0.1)
    whittle.fit(np.log(params), prior=False, approx_grad=True)
    MLEs = whittle.sim_MLEs(params, niter=50, approx_grad=True)
    plot_marginals([MLEs], np.log(params))
    
def dewhittle_full_bayes():
    sim_model = MaternModel()
    sim_model.rho = 10
    sim_model.sigma = 1
    sim_model.nu = 3/2
    sim_model.nugget=0.1
    grid = RectangularGrid((64,64))
    sampler = SamplerOnRectangularGrid(sim_model, grid)
    z = sampler()
    
    niter=5000
    params = np.array([10.,1.])
    model = MaternModel()
    model.nu =3/2
    dw = DeWhittle(z, grid, model, nugget=0.1)
    dw.fit(None, prior=False, approx_grad=False)
    dewhittle_post, A = dw.RW_MH(niter)
    # MLEs = dw.estimate_standard_errors_MLE(np.exp(dw.res.x), monte_carlo=True, niter=500)   # TODO: dont use this method!!
    MLEs = dw.sim_MLEs(params, niter=50)
    dw.prepare_curvature_adjustment()
    adj_dewhittle_post, A = dw.RW_MH(niter, adjusted=True)
    
    title = 'posterior comparisons'
    legend_labels = ['deWhittle', 'adj deWhittle']
    plot_marginals([dewhittle_post, adj_dewhittle_post], np.log(params), title, [r'log$\rho$', r'log$\sigma$'], legend_labels, shape=(1,2))

dewhittle_full_bayes()
whittle_matern_test()
dewhittle_test()
dewhittle_matern_test()
dewhittle_matern_conditional_nu_test()
whittle_test()

# stop
        
# model = MaternModel()
# model.rho = 10
# model.sigma = 1
# model.nu=3/2
# model.nugget=0.1
# grid = RectangularGrid((64,64))
# # print(model(grid.lags_unique))
# sampler = SamplerOnRectangularGrid(model, grid)
# z = sampler()
# params = np.array([10., 1., 1/2])
# model = MaternModel()
# # model.nu=
# dw = DeWhittle(z, grid, model, nugget=0.1)
# dw.fit(np.log(params), prior=False, approx_grad=True)
# MLEs = dw.sim_MLEs(params, niter=50, approx_grad=True)
# plot_marginals([MLEs], np.log(params))

    
# stop
# dewhittle_test(z)