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
from debiased_spatial_whittle.sampling.simulation import SamplerOnRectangularGrid, SamplerOnRectangularGridTapered


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

    def compute_residuals(self, sample = None, model = None):
        if sample is None:
            sample = self.sample
        if model is None:
            model = self.model
        periodogram = self.periodogram_computer(sample)
        ep = ExpectedPeriodogram(self.grid, self.periodogram_computer)(model)
        residuals = 1 - np.exp(-periodogram / ep)
        return residuals

    def spatial_residuals(self, sample = None, model = None):
        randn = BackendManager.get_randn()
        if sample is None:
            sample = self.sample
        if model is None:
            model = self.model
        fftn, ifftn = BackendManager.get_fft_methods()
        periodogram = self.periodogram_computer(sample)
        ep = ExpectedPeriodogram(self.grid, self.periodogram_computer)(model)
        residuals = np.sqrt(periodogram / ep)
        z = randn(*sample.grid.n) + 1j * randn(*sample.grid.n)
        spatial_residuals = fftn(residuals * z) / np.sqrt(sample.grid.n_points)
        return np.real(spatial_residuals)

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
    sampler: SamplerOnRectangularGrid = None,
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
    param_names_repr = model.free_parameters_repr
    # Wrap LaTeX in $...$ for plotly rendering
    param_names = [f'${name}$' for name in param_names_repr]
    
    true_params = model.free_parameters
    n_params = len(true_params)
    
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
    jmat = dbw.jmatrix_sample(model, n_sims=n_sims, sampler=sampler)
    cov_mat = dbw.variance_of_estimates(model, jmat)
    
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


def generate_goodness_of_fit_plots_3d(goodness_of_fit: GoodnessOfFit):
    """
    Generate animated diagnostic plots for 3D GoodnessOfFit data using Plotly.
    
    Creates three plots with animation for the temporal/frame dimension:
    1. Animated image plot of residuals from compute_residuals
    2. Histogram of residuals from compute_residuals (static)
    3. Animated image plot of spatial residuals from spatial_residuals
    
    Parameters
    ----------
    goodness_of_fit : GoodnessOfFit
        GoodnessOfFit object containing the model, grid, and 3D sample data
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure containing the animated diagnostic plots
    """
    # Compute residuals
    residuals = goodness_of_fit.compute_residuals()
    spatial_residuals = goodness_of_fit.spatial_residuals()
    
    # Check if data is 3D
    if residuals.ndim != 3 or spatial_residuals.ndim != 3:
        raise ValueError("generate_goodness_of_fit_plots_3d requires 3D data (n_x, n_y, n_frames)")
    
    # Get number of frames
    n_frames = residuals.shape[-1]
    
    # Create subplot figure for animated plots
    fig = make_subplots(
        rows=1, 
        cols=3,
        subplot_titles=(
            'Residuals Image Plot (Animated)',
            'Residuals Histogram', 
            'Spatial Residuals Image Plot (Animated)'
        ),
        horizontal_spacing=0.15
    )
    
    # Create frames for animation
    frames = []
    
    for frame_idx in range(n_frames):
        # Select current frame
        residuals_frame = residuals[..., frame_idx]
        spatial_residuals_frame = spatial_residuals[..., frame_idx]
        
        # Create traces for this frame
        residuals_img = go.Heatmap(
            z=residuals_frame,
            colorscale='RdBu',
            colorbar=dict(orientation='h', y=-0.4, len=0.25, x=0.15),
            zmin=-2,
            zmax=2
        )
        
        # Histogram of current frame residuals
        residuals_flat = residuals_frame.flatten()
        hist = go.Histogram(
            x=residuals_flat,
            nbinsx=50,
            marker_color='lightblue',
            name='Residuals'
        )
        
        spatial_res_img = go.Heatmap(
            z=spatial_residuals_frame,
            colorscale='RdYlBu',
            colorbar=dict(orientation='h', y=-0.4, len=0.25, x=0.8),
            zmin=-2,
            zmax=2
        )
        
        # Add traces to frame with explicit trace indices
        # This tells Plotly which existing traces to update
        frame_traces = [residuals_img, hist, spatial_res_img]
        frames.append(go.Frame(data=frame_traces, name=str(frame_idx), traces=[0, 1, 2]))
    
    # Add initial frame data (frame 0)
    residuals_frame0 = residuals[..., 0]
    spatial_residuals_frame0 = spatial_residuals[..., 0]
    
    residuals_img_initial = go.Heatmap(
        z=residuals_frame0,
        colorscale='RdBu',
        colorbar=dict(orientation='h', y=-0.4, len=0.25, x=0.15),
        zmin=-2,
        zmax=2
    )
    
    residuals_flat_initial = residuals_frame0.flatten()
    hist_initial = go.Histogram(
        x=residuals_flat_initial,
        nbinsx=50,
        marker_color='lightblue',
        name='Residuals'
    )
    
    spatial_res_img_initial = go.Heatmap(
        z=spatial_residuals_frame0,
        colorscale='RdYlBu',
        colorbar=dict(orientation='h', y=-0.4, len=0.25, x=0.8),
        zmin=-2,
        zmax=2
    )
    
    # Add initial traces
    fig.add_trace(residuals_img_initial, row=1, col=1)
    fig.add_trace(hist_initial, row=1, col=2)
    fig.add_trace(spatial_res_img_initial, row=1, col=3)
    
    # Add animation frames and controls
    fig.frames = frames
    
    # Add animation slider
    sliders = [{
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Frame: ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 300, 'easing': 'cubic-in-out'},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': [{
            'args': [[f.name], {
                'frame': {'duration': 300, 'redraw': False},
                'mode': 'immediate',
                'transition': {'duration': 300}
            }],
            'label': str(frame_idx),
            'method': 'animate'
        } for frame_idx, f in enumerate(frames)]
    }]
    
    fig.update_layout(
        sliders=sliders,
        height=600,
        width=1200,
        title_text='3D Goodness of Fit Diagnostic Plots (Animated)',
        showlegend=False,
        updatemenus=[{
            'type': 'buttons',
            'direction': 'left',
            'pad': {'r': 10, 't': 80},
            'showactive': False,
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top',
            'buttons': [{
                'args': [None, {
                    'frame': {'duration': 500, 'redraw': False},
                    'fromcurrent': True,
                    'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                }],
                'label': '▶️',
                'method': 'animate'
            }, {
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate'
                }],
                'label': '⏸️',
                'method': 'animate'
            }]
        }]
    )
    
    # Update axis labels
    fig.update_xaxes(title_text='X', row=1, col=1)
    fig.update_yaxes(title_text='Y', row=1, col=1)
    
    fig.update_xaxes(title_text='Residual Value', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=2)
    
    fig.update_xaxes(title_text='X', row=1, col=3)
    fig.update_yaxes(title_text='Y', row=1, col=3)
    
    return fig

# Note: For more reliable interactive visualization of 3D data, consider using
# the Dash-based approach in dash_3d_diagnostics.py which provides a slider
# interface for exploring different temporal frames.


def generate_goodness_of_fit_plots(goodness_of_fit: GoodnessOfFit, frame=None):
    """
    Generate diagnostic plots for a GoodnessOfFit object using Plotly.
    
    Creates three plots:
    1. Image plot of residuals from compute_residuals
    2. Histogram of residuals from compute_residuals
    3. Image plot of spatial residuals from spatial_residuals
    
    Parameters
    ----------
    goodness_of_fit : GoodnessOfFit
        GoodnessOfFit object containing the model, grid, and sample
    
    frame : int, optional
        Frame index for 3D data. If None, uses the full data.
        If provided, selects the specified frame from the last dimension.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure containing the three diagnostic plots
    """
    # Compute residuals
    residuals = goodness_of_fit.compute_residuals()
    spatial_residuals = goodness_of_fit.spatial_residuals()
    
    # Handle frame selection for 3D data
    if frame is not None:
        # Select the specified frame from the last dimension
        if residuals.ndim > 2:
            residuals = residuals[..., frame]
        if spatial_residuals.ndim > 2:
            spatial_residuals = spatial_residuals[..., frame]
    
    # Create subplot figure - back to simpler 1-row approach
    fig = make_subplots(
        rows=1, 
        cols=3,
        subplot_titles=(
            'Residuals Image Plot',
            'Residuals Histogram', 
            'Spatial Residuals Image Plot'
        ),
        horizontal_spacing=0.15
    )
    
    # Plot 1: Image plot of residuals
    fftshift, _ = BackendManager.get_fftshift_methods()
    residuals_img = go.Heatmap(
        z=fftshift(residuals),
        colorscale='RdBu',
        colorbar=dict(orientation='h', y=-0.4, len=0.2, x=0.12)
    )
    fig.add_trace(residuals_img, row=1, col=1)
    
    # Plot 2: Histogram of residuals
    residuals_flat = residuals.flatten()
    hist = go.Histogram(
        x=residuals_flat,
        nbinsx=50,
        marker_color='lightblue',
        name='Residuals'
    )
    fig.add_trace(hist, row=1, col=2)
    
    # Plot 3: Image plot of spatial residuals
    spatial_res_img = go.Heatmap(
        z=spatial_residuals,
        colorscale='RdYlBu',
        colorbar=dict(orientation='h', y=-0.4, len=0.2, x=0.88),
        zmin=-2,
        zmax=2
    )
    fig.add_trace(spatial_res_img, row=1, col=3)
    
    # Update layout to make room for horizontal colorbars
    fig.update_layout(
        height=600,
        width=1200,
        title_text='Goodness of Fit Diagnostic Plots',
        showlegend=False,
        margin=dict(b=150)  # Extra bottom margin for colorbars
    )
    
    # Update axis labels
    fig.update_xaxes(title_text='X', row=1, col=1)
    fig.update_yaxes(title_text='Y', row=1, col=1)
    
    fig.update_xaxes(title_text='Residual Value', row=1, col=2)
    fig.update_yaxes(title_text='Count', row=1, col=2)
    
    fig.update_xaxes(title_text='X', row=1, col=3)
    fig.update_yaxes(title_text='Y', row=1, col=3)
    
    return fig