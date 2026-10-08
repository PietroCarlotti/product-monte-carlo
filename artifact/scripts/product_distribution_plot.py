import os

import numpy as np

# Force headless rendering before Matplotlib can initialize the macOS GUI backend.
os.environ["MPLBACKEND"] = "Agg"

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import multivariate_normal

# Typography configuration to match LaTeX serif documents
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
})

# 1. Defining the specific bivariate distribution from your image
mu = np.array([1.0, 2.0])  # Mode located at (1, 2)
cov = np.eye(2)           # Standard covariance matrix
rv = multivariate_normal(mean=mu, cov=cov)

# 2. Domain boundaries
lim = (-2, 5)
x = np.linspace(*lim, 200)
y = np.linspace(*lim, 200)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
Z = rv.pdf(pos)

# 3. Canvas declaration with 3D projection mapping
fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={'projection': '3d'})

# 4. Rendering the primary distribution surface using colorblind-safe 'YlOrRd'
ax.plot_surface(X, Y, Z, cmap='YlOrRd', alpha=0.4, linewidth=0, antialiased=True)

# 5. Vertical slicing plane (Specifically Purple, Semi-transparent)
# We slice along the diagonal y = x, since the product density evaluates
# both components at the same argument (Definition 2.1).
t = np.linspace(*lim, 300)
z_plane = np.linspace(0, Z.max() * 1.3, 50)
T_x, Z_plane_grid = np.meshgrid(t, z_plane)
T_y = T_x  # Diagonal constraint matching the product density's argument

# Purple plane (Color='#9370DB' is MediumPurple)
ax.plot_surface(T_x, T_y, Z_plane_grid, color='#9370DB', alpha=0.15, linewidth=0)

# 6. Intersection curve (Matching solid Purple)
# Evaluate the PDF exactly along the diagonal t_y = t_x
z_curve = rv.pdf(np.column_stack([t, t]))
ax.plot(t, t, z_curve, color='indigo', linewidth=2.5, zorder=10)

# 7. Linear constraint projection/shadow mapped on the Z=0 plane (Dashed Purple)
ax.plot(
    t, t, np.full_like(t, 1e-4), color='indigo', linewidth=2.0,
    alpha=1.0, linestyle='--', zorder=15,
)

# 8. Filled density with black contour isolines on the entire base floor
x_f = np.linspace(*lim, 220)
y_f = np.linspace(*lim, 220)
X_f, Y_f = np.meshgrid(x_f, y_f)
Z_f = rv.pdf(np.dstack((X_f, Y_f)))
floor_colors = plt.cm.YlOrRd(Z_f / Z.max())
floor_colors[..., -1] = 1.0
ax.plot_surface(
    X_f, Y_f, np.zeros_like(Z_f), facecolors=floor_colors,
    linewidth=0, antialiased=False,
)
ax.contour(X_f, Y_f, Z_f, zdir='z', offset=0, levels=8, colors='black', linewidths=1.0, alpha=0.9)

# Pin axis limits to the actual data range to remove matplotlib's default padding
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_zlim(0, Z.max() * 1.3)

# Style the background panes to be transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Style x/y grids to be more visible, keep z grid muted
ax.xaxis._axinfo["grid"]['color'] = (0.3, 0.3, 0.3, 1)
ax.xaxis._axinfo["grid"]['linewidth'] = 1.0
ax.yaxis._axinfo["grid"]['color'] = (0.3, 0.3, 0.3, 1)
ax.yaxis._axinfo["grid"]['linewidth'] = 1.0
ax.zaxis._axinfo["grid"]['color'] = (0.9, 0.9, 0.9, 1)

# Labels
ax.set_xlabel(r'$x_1$', labelpad=8)
ax.set_ylabel(r'$x_2$', labelpad=8)
ax.set_zlabel(r'$f(x_1, x_2)$', labelpad=8)

# View initialization and proportions
ax.view_init(elev=28, azim=-55)
ax.set_box_aspect([1, 1, 0.6])
ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

plt.tight_layout()
out = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..',
    'figures',
    'product_distribution.pdf',
)
plt.savefig(out, bbox_inches='tight', dpi=150)
print(f'Saved to {out}')
