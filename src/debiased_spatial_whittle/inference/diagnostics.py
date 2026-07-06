from debiased_spatial_whittle.backend import BackendManager
np = BackendManager.get_backend()

from functools import cached_property
from scipy.stats import chisquare, norm, chi2
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from debiased_spatial_whittle.models.base import CovarianceModel, ModelInterface
from debiased_spatial_whittle.grids.base import RectangularGrid
from debiased_spatial_whittle.inference.periodogram import Periodogram, ExpectedPeriodogram
from debiased_spatial_whittle.inference.multivariate_periodogram import Periodogram as MultivariatePeriodogram
from debiased_spatial_whittle.inference.likelihood import DebiasedWhittle, MultivariateDebiasedWhittle, Estimator
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid


class GoodnessOfFit:
    """
    Class to perform a goodness of fit analysis between a model and a sampled random field.
    """
    def __init__(
        self, model: CovarianceModel, grid: RectangularGrid, sample, n_bins: int = 10
    ):
        """
        Parameters
        ----------
        model
            Covariance model for the data
        grid
            Sampling grid
        sample
            sampled random field data
        n_bins
            Number of bins used in the goodness-of-fit analysis
        """
        self.model = model
        self.grid = grid
        self.sample = sample
        self.n_bins = n_bins
        self.periodogram_computer = Periodogram()
        self.bootstrap = True

    @cached_property
    def sampler(self):
        return SamplerOnRectangularGrid(self.model, self.grid)

    def compute_residuals(self, sample, model):
        periodogram = self.periodogram_computer(sample)
        ep = ExpectedPeriodogram(self.grid, self.periodogram_computer)(model)
        residuals = 1 - np.exp(-periodogram / ep)
        return residuals

    def compute_diagnostic_statistic(self, sample=None, model=None):
        if sample is None:
            sample = self.sample
            model = self.model
        residuals = self.compute_residuals(sample, model).flatten()
        bin_counts = np.bincount((residuals * self.n_bins).astype(np.int64))
        statistic, pvalue = chisquare(bin_counts)
        return statistic, pvalue

    def p_value(self, statistic: float, n_sim: int = 20):
        statistic_values = []
        dbw = DebiasedWhittle(
            self.periodogram_computer,
            ExpectedPeriodogram(self.grid, self.periodogram_computer),
        )
        estimator = Estimator(dbw)
        for i in range(n_sim):
            sample = self.sampler()
            if self.bootstrap:
                model_est = self.get_model_est()
                estimator(model_est, sample)
                statistic_value, _ = self.compute_diagnostic_statistic(
                    sample, model_est
                )
            else:
                statistic_value, _ = self.compute_diagnostic_statistic(
                    sample, self.model
                )
            statistic_values.append(statistic_value)
        return np.mean(statistic <= statistic_values)


def corner_plot_variance_of_estimates(
    model: ModelInterface,
    grid: RectangularGrid,
    dbw: DebiasedWhittle | MultivariateDebiasedWhittle = None,
    n_sims: int = 250,
    width: int = 800,
    height: int = 800,
    **kwargs
):
    """
    Create a corner plot showing the distribution of parameter estimates using Plotly.
    
    On the diagonal, shows histograms of normal distributions with mean equal to
    the true parameter values and variance from the covariance matrix.
    On the off-diagonal, shows 95% confidence ellipses of bivariate normal distributions.
    
    Parameters
    ----------
    model : ModelInterface
        Covariance model with true parameter values
    grid : RectangularGrid
        The grid used for the analysis
    dbw : DebiasedWhittle or MultivariateDebiasedWhittle, optional
        The DebiasedWhittle instance. If None, it will be created.
    n_sims : int, default=250
        Number of simulations for computing jmatrix_sample if dbw is None
    width : int, default=800
        Width of the figure in pixels
    height : int, default=800
        Height of the figure in pixels
    **kwargs
        Additional keyword arguments passed to the subplot titles
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The figure object containing the corner plot
    """
    # Get parameter names and true values
    # Use free_parameters_repr if available for LaTeX representations
    if hasattr(model, 'free_parameters_repr'):
        param_names_raw = model.parameter_names
        param_names_repr = model.free_parameters_repr
        # Wrap LaTeX in $...$ for plotly rendering
        param_names = [f'${name}$' for name in param_names_repr]
    else:
        param_names = model.parameter_names
        param_names_raw = model.parameter_names
    
    true_params = model.get_parameters(param_names_raw)
    n_params = len(param_names)
    
    # Convert to numpy arrays
    import numpy as np
    true_params_np = np.asarray([
        p.numpy() if hasattr(p, 'numpy') else 
        (p.cpu().numpy() if hasattr(p, 'cpu') else p) 
        for p in true_params
    ], dtype=np.float64)
    
    # Create dbw if not provided
    if dbw is None:
        # Determine if we need multivariate version based on grid.nvars
        if grid.nvars > 1:
            periodogram = MultivariatePeriodogram()
            ep = ExpectedPeriodogram(grid, periodogram)
            dbw = MultivariateDebiasedWhittle(periodogram, ep)
        else:
            periodogram = Periodogram()
            ep = ExpectedPeriodogram(grid, periodogram)
            dbw = DebiasedWhittle(periodogram, ep)
    
    # Compute variance of estimates
    cov_mat = dbw.variance_of_estimates(model)
    
    # Convert to numpy if needed (for torch backend)
    if hasattr(cov_mat, 'numpy'):
        cov_mat = cov_mat.numpy()
    elif hasattr(cov_mat, 'cpu'):
        cov_mat = cov_mat.cpu().numpy()

    # Compute standard deviations from covariance matrix
    std_devs = np.sqrt(np.diag(cov_mat))
    
    # Create subplot grid without titles
    fig = make_subplots(
        rows=n_params,
        cols=n_params,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
    )
    
    # Create corner plot
    for i in range(n_params):
        for j in range(n_params):
            row = i + 1
            col = j + 1
            
            if i == j:
                # Diagonal: PDF of normal distribution
                mu = true_params_np[i]
                sigma = std_devs[i]
                
                # Generate x values for PDF
                x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
                pdf = norm.pdf(x, mu, sigma)
                
                # Add PDF curve
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=pdf,
                        mode='lines',
                        fill='tozeroy',
                        name=f'{param_names[i]}',
                        line=dict(color='skyblue', width=2),
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )
                
                # Add true parameter line
                fig.add_vline(
                    x=true_params_np[i],
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"True: {true_params_np[i]:.4f}",
                    row=row,
                    col=col,
                )
                
                # Update axis labels - only on outer edges
                # For diagonal (histograms), show x-axis label only on bottom row
                if i == n_params - 1:
                    fig.update_xaxes(title_text=param_names[i], row=row, col=col)
                else:
                    fig.update_xaxes(title_text="", row=row, col=col)
                
                # For leftmost column, show parameter name on y-axis
                if j == 0:
                    fig.update_yaxes(title_text=param_names[i], row=row, col=col)
                else:
                    fig.update_yaxes(title_text="", row=row, col=col)
                
            elif i < j:
                # Upper triangle: empty - keep ticks but hide labels
                fig.update_xaxes(title_text="", showticklabels=False, row=row, col=col)
                fig.update_yaxes(title_text="", showticklabels=False, row=row, col=col)
                
            else:
                # Lower triangle: confidence ellipse
                # For the lower triangle (i > j), x-axis is parameter j, y-axis is parameter i
                
                # Compute 95% confidence ellipse
                # For bivariate normal, use chi-squared value for 95% confidence
                chi2_95 = chi2.ppf(0.95, 2)
                
                # Get sub-covariance matrix for parameters j (x) and i (y)
                # cov_ji = [[var(j), cov(j,i)], [cov(i,j), var(i)]]
                cov_ji = np.array([
                    [cov_mat[j, j], cov_mat[j, i]],
                    [cov_mat[i, j], cov_mat[i, i]]
                ])
                
                # Compute eigenvalues and eigenvectors
                eigvals, eigvecs = np.linalg.eigh(cov_ji)
                
                # Sort in descending order
                idx = eigvals.argsort()[::-1]
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]
                
                # Compute angle of rotation
                angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                
                # Compute width and height of ellipse
                ellipse_width = 2 * np.sqrt(chi2_95 * eigvals[0])
                ellipse_height = 2 * np.sqrt(chi2_95 * eigvals[1])
                
                # Create ellipse using parametric equations
                theta = np.linspace(0, 2 * np.pi, 100)
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                angle_rad = np.radians(angle)
                cos_angle = np.cos(angle_rad)
                sin_angle = np.sin(angle_rad)
                
                # Center at (mu_j, mu_i) since x=j, y=i
                ellipse_x = (
                    true_params_np[j] + 
                    ellipse_width / 2 * cos_theta * cos_angle - 
                    ellipse_height / 2 * sin_theta * sin_angle
                )
                ellipse_y = (
                    true_params_np[i] + 
                    ellipse_width / 2 * cos_theta * sin_angle + 
                    ellipse_height / 2 * sin_theta * cos_angle
                )
                
                # Add ellipse trace
                fig.add_trace(
                    go.Scatter(
                        x=ellipse_x,
                        y=ellipse_y,
                        mode='lines',
                        line=dict(color='red', width=2),
                        name=f'95% CI',
                        showlegend=(i == n_params - 1 and j == 0),
                    ),
                    row=row,
                    col=col,
                )
                
                # Mark true parameter value
                fig.add_trace(
                    go.Scatter(
                        x=[true_params_np[j]],
                        y=[true_params_np[i]],
                        mode='markers',
                        marker=dict(color='red', size=8, symbol='x'),
                        name='True value',
                        showlegend=(i == n_params - 1 and j == 0),
                    ),
                    row=row,
                    col=col,
                )
                
                # Update axis labels - only on outer edges
                # For lower triangle, only show x-axis label on bottom row
                if i == n_params - 1:
                    fig.update_xaxes(title_text=param_names[j], row=row, col=col)
                else:
                    fig.update_xaxes(title_text="", row=row, col=col)
                
                # Only show y-axis label on leftmost column
                if j == 0:
                    fig.update_yaxes(title_text=param_names[i], row=row, col=col)
                else:
                    fig.update_yaxes(title_text="", row=row, col=col)
    
    # Update layout
    fig.update_layout(
        width=int(width),
        height=int(height),
        title_text="Corner Plot: Parameter Estimate Distributions",
        showlegend=True,
        legend=dict(x=1.05, y=1.0),
    )
    
    return fig