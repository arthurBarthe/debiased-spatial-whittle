from debiased_spatial_whittle.backend import BackendManager
from debiased_spatial_whittle.grids.base import RectangularGrid, FourierGrid
from debiased_spatial_whittle.caching import Freezable, ban_if_frozen

xp = BackendManager.get_backend()


class Sample(Freezable):
    """
    General class for the definition of a sampled random field. Allows to store computed quantities such as
    periodograms etc.
    """

    def __init__(self, grid: RectangularGrid, values: xp.ndarray):
        self.grid = grid
        self.values = values
        super().__init__()
        self.freeze()

    def __array__(self, *args, **kwargs):
        return xp.to_numpy(self.values)


class SampleOnRectangularGrid(Sample):
    """
    Class for a sample on a Rectangular grid. In the case of a grid with missing observations, the values at
    missing locations are not used.
    """

    def __init__(self, grid: RectangularGrid, values: xp.ndarray):
        super(SampleOnRectangularGrid, self).__init__(grid, values)
        assert isinstance(
            grid, RectangularGrid
        ), "The grid should be an instance of RectangularGrid"
        if self.grid.nvars == 1:
            assert values.shape == grid.n, "The shape of the values does not match the grid"

    def show(self, vmin=None, vmax=None):
        """
        Visualize the sample data.
        For 1D data: displays a line plot.
        For 2D data: displays a simple heatmap.
        For 3D data: displays an animated heatmap.

        Parameters
        ----------
        vmin : float, optional
            Minimum value for the color scale in heatmaps.
        vmax : float, optional
            Maximum value for the color scale in heatmaps.
        """
        import plotly.graph_objects as go
        import numpy as np

        values_np = xp.to_numpy(self.values)
        values_np = np.where(values_np == 0, np.nan, values_np)
        extent = self.grid.extent

        if values_np.ndim == 1:
            # Line plot for 1D data
            x = np.arange(values_np.shape[0]) * self.grid.delta[0]
            fig = go.Figure(data=go.Scatter(x=x, y=values_np, mode='lines'))
            fig.update_layout(
                width=400,
                height=400,
                xaxis_title="Position",
                yaxis_title="Value"
            )
            fig.update_xaxes(range=[0, extent[0]])
            fig.show()
        elif values_np.ndim == 2:
            # Simple heatmap for 2D data
            fig = go.Figure(data=go.Heatmap(z=values_np, zmin=vmin, zmax=vmax))
            fig.update_layout(
                width=400,
                height=400
            )
            fig.update_xaxes(range=[0, extent[0]])
            fig.update_yaxes(range=[0, extent[1]])
            fig.show()
        elif values_np.ndim == 3:
            # Animated heatmap for 3D data
            N = values_np.shape[-1]
            zmin = vmin if vmin is not None else np.min(values_np)
            zmax = vmax if vmax is not None else np.max(values_np)
            fig = go.Figure(
                data=go.Heatmap(z=values_np[..., 0], zmin=zmin, zmax=zmax),
                frames=[go.Frame(data=go.Heatmap(z=values_np[..., i], zmin=zmin, zmax=zmax)) for i in range(N)]
            )
            fig.update_layout(
                width=400,
                height=400,
                updatemenus=[
                    dict(type="buttons", visible=True,
                         buttons=[dict(label="Play", method="animate",
                                     args=[None, {"frame": {"duration": 50, "redraw": True}}])]
                         )
                ]
            )
            fig.update_xaxes(range=[0, extent[0]])
            fig.update_yaxes(range=[0, extent[1]])
            fig.show()
        else:
            raise ValueError(f"Visualization not supported for {values_np.ndim}D data. Only 1D, 2D and 3D are supported.")
        return fig


class FourierValues(Sample):
    """
    Class for Fourier-transformed values on a FourierGrid.
    The show method applies fftshift and 10 * log10 transformation to the data for proper visualization.
    """

    def __init__(self, grid: FourierGrid, values: xp.ndarray):
        super(FourierValues, self).__init__(grid, values)
        assert isinstance(
            grid, FourierGrid
        ), "The grid should be an instance of FourierGrid"

    def show(self, vmin=None, vmax=None):
        """
        Visualize the sample data with fftshift and 10 * log10 transformation applied.
        For 1D data: displays a line plot.
        For 2D data: displays a simple heatmap.
        For 3D data: displays an animated heatmap.

        Parameters
        ----------
        vmin : float, optional
            Minimum value for the color scale in heatmaps.
        vmax : float, optional
            Maximum value for the color scale in heatmaps.
        """
        import plotly.graph_objects as go
        import numpy as np

        values_np = xp.to_numpy(self.values)
        # Replace zeros with nan before log transform
        values_np = np.where(values_np == 0, np.nan, values_np)
        # Apply 10 * log10 transformation
        values_np = 10 * np.log10(values_np)
        # Apply fftshift for proper Fourier space visualization
        values_np = np.fft.fftshift(values_np, axes=tuple(range(values_np.ndim)))
        
        # Calculate shifted extent: centered at zero
        extent = self.grid.extent
        extent_half = tuple([e / 2 for e in extent])

        if values_np.ndim == 1:
            # Line plot for 1D data
            n = values_np.shape[0]
            # Create frequency axis (fftshifted)
            freq = np.fft.fftshift(np.fft.fftfreq(n, d=self.grid.original_grid.delta[0]))
            fig = go.Figure(data=go.Scatter(x=freq, y=values_np, mode='lines'))
            fig.update_layout(
                width=400,
                height=400,
                xaxis_title="Frequency",
                yaxis_title="10 * log10(Value)"
            )
            fig.update_xaxes(range=[-extent_half[0], extent_half[0]])
            fig.show()
        elif values_np.ndim == 2:
            # Simple heatmap for 2D data
            # Create fftshifted coordinate arrays
            x = np.fft.fftshift(np.fft.fftfreq(values_np.shape[1], d=self.grid.original_grid.delta[1]))
            y = np.fft.fftshift(np.fft.fftfreq(values_np.shape[0], d=self.grid.original_grid.delta[0]))
            fig = go.Figure(data=go.Heatmap(z=values_np, x=x, y=y, zmin=vmin, zmax=vmax))
            fig.update_layout(
                width=400,
                height=400
            )
            fig.show()
        elif values_np.ndim == 3:
            # Animated heatmap for 3D data
            N = values_np.shape[-1]
            # Create fftshifted coordinate arrays
            x = np.fft.fftshift(np.fft.fftfreq(values_np.shape[1], d=self.grid.original_grid.delta[1]))
            y = np.fft.fftshift(np.fft.fftfreq(values_np.shape[0], d=self.grid.original_grid.delta[0]))
            zmin = vmin if vmin is not None else np.nanmin(values_np)
            zmax = vmax if vmax is not None else np.nanmax(values_np)
            fig = go.Figure(
                data=go.Heatmap(z=values_np[..., 0], x=x, y=y, zmin=zmin, zmax=zmax),
                frames=[go.Frame(data=go.Heatmap(z=values_np[..., i], x=x, y=y, zmin=zmin, zmax=zmax)) for i in range(N)]
            )
            fig.update_layout(
                width=400,
                height=400,
                updatemenus=[
                    dict(type="buttons", visible=True,
                         buttons=[dict(label="Play", method="animate", 
                                     args=[None, {"frame": {"duration": 50, "redraw": True}}])]
                         )
                ]
            )
            fig.show()
        else:
            raise ValueError(f"Visualization not supported for {values_np.ndim}D data. Only 1D, 2D and 3D are supported.")
        return fig
