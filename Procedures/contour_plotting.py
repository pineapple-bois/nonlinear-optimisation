import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_contour(f_func, f_symbolic, x_range, y_range,
                 levels=50, cmap='viridis', title=None):
    """
    Plots a contour plot for a given function.

    Parameters
    ----------
    f_func : callable
        Lambdified function to evaluate on the grid.
    f_symbolic : sympy.Expr
        Symbolic representation of the function (for the title).
    x_range : tuple
        Tuple specifying the range for the x-axis (xmin, xmax).
    y_range : tuple
        Tuple specifying the range for the y-axis (ymin, ymax).
    levels : int, optional
        Number of contour levels for the plot (default is 50).
    cmap : str, optional
        Colormap for the contour plot (default is 'viridis').
    title : str, optional
        Custom title for the plot. If None, the title is generated from the function.
    """
    # Generate a meshgrid over the specified range
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate the function on the grid
    Z = f_func(X, Y)

    # Create the contour plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, levels=levels, cmap=cmap)
    plt.colorbar(contour, label='Function Value')  # Add colorbar with label
    plt.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5)  # Overlay contour lines

    # Add axis labels
    plt.xlabel('x')
    plt.ylabel('y')

    # Set the title
    if title is None:
        title = f"Contour Plot\n$f(x, y) = {sp.latex(f_symbolic)}$"
    plt.title(title)

    # Show the plot
    plt.show()


def plot_clamped_contour(f_func, f_symbolic, x_range, y_range,
                         minimisers=None, clamp=0.5,
                         levels=20, cmap='viridis'):
    """
    Plots a clamped contour plot for a given function with optional minimisers overlay.

    Parameters
    ----------
    f_func : callable
        Lambdified function to evaluate on the grid.
    f_symbolic : sympy.Expr
        Symbolic representation of the function (for the title).
    x_range : tuple
        Tuple specifying the range for the x-axis (xmin, xmax).
    y_range : tuple
        Tuple specifying the range for the y-axis (ymin, ymax).
    minimisers : list or tuple, optional
        List of minimisers (or a single minimiser) to overlay on the plot. Defaults to None.
    clamp : float, optional
        Value to clamp the function output. Defaults to 0.5.
    levels : int, optional
        Number of contour levels for the plot. Defaults to 20.
    cmap : str, optional
        Colormap for the contour plot. Defaults to 'viridis'.
    """
    # Generate a meshgrid over the specified range
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate the function on the grid
    Z = f_func(X, Y)

    # Clamp Z values for better visualisation
    clamped_Z = np.minimum(Z, np.min(Z) + clamp)

    # Create the contour plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, clamped_Z, levels=levels, cmap=cmap)
    plt.colorbar(contour, label='Function Value (Clamped)')  # Add labelled colorbar
    plt.contour(X, Y, clamped_Z, levels=10, colors='black', linewidths=0.5)  # Overlay contour lines

    # Overlay minimisers if provided
    if minimisers is not None:
        if not isinstance(minimisers, list):
            minimisers = [minimisers]  # Convert a single minimiser into a list
        # Generate unique colours for each minimiser
        colours = plt.cm.tab10(np.linspace(0, 1, len(minimisers)))
        for i, minimiser in enumerate(minimisers):
            plt.plot(
                minimiser[0], minimiser[1],
                marker='o',
                color=colours[i],
                label=f"{i + 1}: {np.round(minimiser, 4)}"
            )
        plt.legend()

    # Add axis labels and title
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title(f'Contour Plot with Clamped Values\n$f(x, y) = {sp.latex(f_symbolic)}$', fontsize=14)

    # Show the plot
    plt.show()


def plot_surface(f_func, f, x_range, y_range, elev=30, azim=-60):
    """
    Generate a 3D surface plot for a given function f(x, y).

    Parameters:
    ----------
    f_func : function
        Lambdified function to evaluate f(x, y) on the grid.
    f : sympy.Expr
        Original symbolic expression for displaying in the title.
    x_range : tuple
        Range for x-axis as (min, max).
    y_range : tuple
        Range for y-axis as (min, max).
    elev : int, optional
        Elevation angle for the plot's view (default is 30 degrees).
    azim : int, optional
        Azimuthal angle for the plot's view (default is -60 degrees).

    Returns:
    -------
    None
    """
    # Generate a dense meshgrid for the plot
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate f on the grid
    Z = f_func(X, Y)

    # Create the 3D surface plot
    fig = plt.figure(figsize=(12, 10))
    axes = fig.add_subplot(projection='3d')

    # Plot the surface
    surf = axes.plot_surface(X, Y, Z, cmap='jet', edgecolor='none', alpha=0.9)

    # Add a colour bar for reference
    fig.colorbar(surf, ax=axes, shrink=0.5, aspect=10, label='Function Value')

    # Set the viewing direction
    axes.view_init(elev=elev, azim=azim)

    # Add axis labels and title
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('f(x, y)')
    axes.set_title(f'3D Surface Plot\n$f(x, y) = {{{sp.latex(f)}}}$', fontsize=14)

    # Add grid lines for clarity
    axes.grid(True)

    # Show the plot
    plt.show()
