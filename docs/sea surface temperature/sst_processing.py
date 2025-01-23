from time import time
from PIL import Image
from scipy import stats
import autograd.numpy as np

# import statsmodels.api as sm
import matplotlib.pyplot as plt
from autograd.scipy import stats
from autograd.scipy import special
from scipy.special import kv as BesselK  # Modified Bessel of second kind
from autograd import grad, hessian, jacobian


lat = np.loadtxt("latitude.txt")
long = np.loadtxt("longitude.txt")
sst = np.loadtxt("sst_raw.txt")  # raw data
# sst = (sst-sst.mean())/sst.std()

n1, n2 = sst.shape
N = n1 * n2


lat_range = [20, 38.875]
long_range = [211.125, 230]


lat = lat[(lat_range[0] < lat) & (lat < lat_range[1])]
long = long[(long_range[0] < long) & (long < long_range[1])]
long_grid, lat_grid = np.meshgrid(long, lat)


Xlong = long_grid.ravel()
Xlat = lat_grid.ravel()
Xlong2 = Xlong**2
Xlat2 = Xlat**2
Xlonglat = Xlong * Xlat

X = np.vstack((np.ones(n1 * n2), Xlong, Xlat, Xlong2, Xlat2, Xlonglat)).T
# print(X)

y = sst.ravel()
Bhat = np.linalg.inv(X.T @ X).T @ X.T @ y

data = (y - np.sum(Bhat * X, axis=1)).reshape(
    n1, n2
)  # data are now residuals, removing second-order polynomial trend
# data = (data-data.mean())/data.std()

fig, ax = plt.subplots(1, 2, figsize=(20, 15))
ax[0].set_title("Sea Temperature Data", fontsize=22)
im1 = ax[0].imshow(
    sst, cmap="binary", origin="lower", extent=[211.125, 230, 20, 38.875]
)
fig.colorbar(im1, shrink=0.5, ax=ax[0])

ax[1].set_title("Residuals", fontsize=22)
im2 = ax[1].imshow(
    data, cmap="binary", origin="lower", extent=[211.125, 230, 20, 38.875]
)
fig.colorbar(im2, shrink=0.5, ax=ax[1])

for i in range(2):
    ax[i].set_xlabel("Longitude", fontsize=16)
    ax[i].set_ylabel("Latitude", fontsize=16)

    ax[i].set_xticks(np.arange(215, 235, 5))
    ax[i].set_yticks(np.arange(20, 38.875, 5))

fig.tight_layout()
plt.show()


# def MaternSpecDens(omega,params):

#     d=2
#     l,v,sigma2 = params
#     phi = special.gamma(v+d/2)*(4*v)**v/(np.pi**(d/2)*l**(2*v)*special.gamma(v))
#     f   = sigma2*phi/((4*v)/l**2 + omega**2)**(v+d/2)
#     return f  # /(2*np.pi)**2  # + nugget  /(2*np.pi)**2


def MaternSpectralDens(F1, F2, params, inf_sum=False):
    """
    Infinite sum version of 2-d Spectral density of Matern kernel due to aliasing defined in Fuentes paper (Handcock and Wallis 1994)
    F1,F2 are (n1,n2) grids
    """

    d = 2
    n1, n2 = F1.shape
    N = n1 * n2

    l, v, sigma2 = params
    nugget = 0.000

    if inf_sum:
        grid1 = 2 * np.pi * np.arange(-5, 5)  # /(delta[0])
        grid2 = 2 * np.pi * np.arange(-5, 5)  # /(delta[1])
        G1, G2 = np.meshgrid(grid1, grid2)

        arg1 = np.tile(G1, (n1 * n2, 1, 1)) + F1.ravel()[:, None, None]
        arg2 = np.tile(G2, (n1 * n2, 1, 1)) + F2.ravel()[:, None, None]

        omega2 = arg1**2 + arg2**2  # .flatten()[:N//2]

    else:
        omega2 = (F1**2 + F2**2).flatten()[: N // 2]

    phi = (
        special.gamma(v + d / 2)
        * (4 * v) ** v
        / (np.pi ** (d / 2) * l ** (2 * v) * special.gamma(v))
    )
    f = sigma2 * phi / ((4 * v) / l**2 + omega2) ** (v + d / 2)

    if inf_sum:
        return (
            np.sum(f, axis=(1, 2))[: N // 2] + nugget / ((2 * np.pi) ** 2)
        )  # np.sum(f, axis=(1,2)).reshape(n1,n2)# .flatten()[:N//2] # #   # /(2*np.pi)**2  # + nugget/((2*np.pi)**2)

    return f + nugget / ((2 * np.pi) ** 2)  # /(2*np.pi)**2 #


def whittle_2d(params, F1, F2, I, inf_sum=False, return_sum=False):
    params = np.exp(params)
    f = MaternSpectralDens(F1, F2, params, inf_sum=inf_sum)

    loglik = -(N / (2 * np.pi) ** 2) * (np.log(f) + I / f)
    if return_sum:
        return np.sum(loglik)

    return loglik


delta = np.array([27.83, 32.4])
# delta = np.array([.25,.25])
# delta = np.ones(2)

I = (
    np.prod(delta)
    * np.abs(np.fft.fftshift(np.fft.fft2(data))) ** 2
    / (4 * N * np.pi**2)
)  # pgram

f1 = 2 * np.pi * np.arange(-n1 // 2, n1 // 2 + 1) / (n1 * delta[0])  # +1?
f2 = 2 * np.pi * np.arange(-n2 // 2, n2 // 2 + 1) / (n2 * delta[1])  # +1?

f1 = f1[f1 != 0]
f2 = f2[f2 != 0]

F1, F2 = np.meshgrid(f1, f2)
F = np.stack((F1, F2)).reshape(2, N).T


inf_sum = False
params = np.log([264.6, 0.425, 0.0033])  # np.sqrt(0.0033)


def whittle(x):
    return whittle_2d(x, F1, F2, I.flatten()[: N // 2], inf_sum, return_sum=True)


logpost = lambda x: whittle(x)  # + logprior(x)
# print(logpost(true_params))


def obj(prm):
    return -logpost(prm)


jacob = grad(obj)  # jacobian = gradian for scalar valued functions
gradll, hessll = grad(logpost), hessian(logpost)


from scipy.optimize import minimize

MAP = minimize(obj, jac=jacob, hess=hessll, method="trust-constr", x0=params)
print(f'Optimizer convergence: {MAP["success"]} \n')
print(f"Used infinite sum: {inf_sum}")
print(f"MAP         :  {np.round(np.exp(MAP.x),3)}")
# print(f'true params :  {np.round(np.exp(params),3)} \n')
print(f"log post at MAP:        {np.round(-MAP.fun,3)}")
# print(f'log post at true theta: {np.round(logpost(params),3)} \n')
# stop


def Matern_infsum(omega2, params):
    """
    Infinite sum version of 2-d Spectral density of Matern kernel due to aliasing defined in Fuentes paper (Handcock and Wallis 1994)
    F1,F2 are (n1,n2) grids
    """

    d = 2
    l, v, sigma2 = params

    # omega2 = (w1+G1)**2 + (w2+G2)**2

    phi = (
        special.gamma(v + d / 2)
        * (4 * v) ** v
        / (np.pi ** (d / 2) * l ** (2 * v) * special.gamma(v))
    )
    f = sigma2 * phi / ((4 * v) / l**2 + omega2) ** (v + d / 2)

    return f.sum()  # /(2*np.pi)**2 #  + nugget/((2*np.pi)**2)


omega = np.sqrt(F1**2 + F2**2).flatten()[: N // 2]
# omega_grid = np.unique(omega)

grid1 = 2 * np.pi * np.arange(-5, 5)  # /(delta[0])
grid2 = 2 * np.pi * np.arange(-5, 5)  # /(delta[1])
# grid1 = 2*np.pi*np.arange(1,11)#/(delta[0])
# grid2 = 2*np.pi*np.arange(1,11)#/(delta[1])
G1, G2 = np.meshgrid(grid1, grid2)

F = F[: N // 2]
f_infsum = np.array(
    [Matern_infsum((w[0] + G1) ** 2 + (w[1] + G2) ** 2, np.exp(MAP.x)) for w in F]
)


f = MaternSpectralDens(F1, F2, np.exp(MAP.x), inf_sum)

plt.figure(figsize=(15, 7))
plt.title(f"Infinite sum: {inf_sum}", fontsize=16)
plt.plot(omega, np.log(I.flatten()[: N // 2]), ".")
plt.plot(omega[: N // 2], np.log(f), label=r"$f(\omega)$")
plt.plot(omega[: N // 2], np.log(f_infsum), label=r"$f_{\Delta}(\omega)$")
plt.legend(fontsize=18)
plt.show()

# plt.figure(figsize=(15,7))
# plt.plot(omega,f.ravel()[:N//2], '--')


# fig = plt.figure(figsize=(15,15))
# ax = plt.axes(projection='3d')
# # ax.contour3D(F1,F2,a, 50, cmap='viridis')
# # im = ax.scatter(F[:,0], F[:,1], I, c=I, cmap = 'autumn_r', s=10)
# # ax.contour3D(F1, F2, f_infsum, levels=20, cmap='Reds', linestyles="solid")
# ax.contour(F1, F2, f, levels=20, cmap='binary', linestyles="solid")
# # fig.colorbar(im, fraction=0.03)
# plt.show()
