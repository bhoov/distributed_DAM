import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# set the colormap and centre the colorbar
class MidpointNormalize(mpl.colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

cnorm=MidpointNormalize(midpoint=0.)

def show_img(img, ax=None, fig=None, axis_off=True, aspect='equal', show_colorbar=True, **kwargs):
    vmin, vmax = img.min(), img.max()
    vscale = max(np.abs(vmin), np.abs(vmax))
    cnorm = MidpointNormalize(midpoint=0., vmin=-vscale, vmax=vscale)

    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(1,1, **kwargs)
        else:
            ax = fig.subplots(1,1, **kwargs)

    pcm = ax.imshow(img, cmap="seismic", norm=cnorm, aspect=aspect)

    if axis_off:
        ax.axis("off")

    if show_colorbar:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.83, 0.15, 0.03, 0.7])
        fig.colorbar(pcm, cax=cbar_ax);
    return fig, ax;
