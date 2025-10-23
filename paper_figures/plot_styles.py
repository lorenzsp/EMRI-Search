"""
Matplotlib style configuration for consistent plotting across scripts.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from contextlib import contextmanager
import numpy as np

# Colorblind-friendly color palettes
COLORBLIND_PALETTES = {
    # Tol's bright qualitative scheme (up to 7 colors)
    'tol_bright': ['#EE6677', '#228833', '#4477AA', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB'],
    
    # Tol's muted qualitative scheme (up to 9 colors)  
    'tol_muted': ['#CC6677', '#332288', '#DDCC77', '#117733', '#88CCEE', '#882255', '#44AA99', '#999933', '#AA4499'],
    
    # Tol's light qualitative scheme (up to 9 colors)
    'tol_light': ['#BBCC33', '#AAAA00', '#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF', '#44BB99', '#DDDDDD'],
    
    # Okabe-Ito palette (most common colorblind-friendly palette)
    'okabe_ito': ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000'],
    
    # Viridis-like discrete colors
    'viridis_discrete': ['#440154', '#31688e', '#35b779', '#fde725'],
    
    # Wong palette (good for presentations)
    'wong': ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'],
}

default_palette = 'tol_bright'

def get_colorblind_palette(name=default_palette, n_colors=None):
    """
    Get a colorblind-friendly color palette.
    
    Parameters:
    -----------
    name : str
        Name of the palette ('okabe_ito', 'tol_bright', 'tol_muted', 'tol_light', 'viridis_discrete', 'wong')
    n_colors : int, optional
        Number of colors to return. If None, returns all colors in palette.
    
    Returns:
    --------
    list : List of hex color codes
    """
    palette = COLORBLIND_PALETTES.get(name, COLORBLIND_PALETTES['okabe_ito'])
    if n_colors is not None:
        palette = palette[:n_colors]
    return palette

def set_colorblind_cycle(name=default_palette):
    """Set the default color cycle to a colorblind-friendly palette."""
    colors = get_colorblind_palette(name)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

# Physical Review style settings
PHYSREV_STYLE = {
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.figsize': (6.5, 3.7),
    'savefig.dpi': 300,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.8,
}

def apply_physrev_style(colorblind=True, palette=default_palette):
    """Apply Physical Review style to matplotlib."""
    plt.rcParams.update(PHYSREV_STYLE)
    if colorblind:
        set_colorblind_cycle(palette)

def apply_style(style_dict):
    """Apply custom style dictionary to matplotlib."""
    plt.rcParams.update(style_dict)

def reset_style():
    """Reset matplotlib to default style."""
    plt.rcdefaults()

@contextmanager
def physrev_style(colorblind=True, palette=default_palette):
    """Context manager for temporary style application."""
    old_params = plt.rcParams.copy()
    try:
        apply_physrev_style(colorblind=colorblind, palette=palette)
        yield
    finally:
        plt.rcParams.update(old_params)

# Alternative style configurations
PRESENTATION_STYLE = {
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'legend.fontsize': 16,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'figure.figsize': (10, 6),
    'savefig.dpi': 300,
    'lines.linewidth': 2.0,
    'axes.linewidth': 1.2,
}

def apply_presentation_style(colorblind=True, palette=default_palette):
    """Apply presentation style to matplotlib."""
    plt.rcParams.update(PRESENTATION_STYLE)
    if colorblind:
        set_colorblind_cycle(palette)

@contextmanager
def presentation_style(colorblind=True, palette=default_palette):
    """Context manager for temporary presentation style application."""
    old_params = plt.rcParams.copy()
    try:
        apply_presentation_style(colorblind=colorblind, palette=palette)
        yield
    finally:
        plt.rcParams.update(old_params)

# Utility functions for colorblind-friendly plotting
def create_colorblind_cmap(name=default_palette, n_colors=256):
    """
    Create a colorblind-friendly colormap.
    
    Parameters:
    -----------
    name : str
        Base palette name
    n_colors : int
        Number of colors in the colormap
        
    Returns:
    --------
    matplotlib.colors.LinearSegmentedColormap
    """
    colors = get_colorblind_palette(name)
    return mcolors.LinearSegmentedColormap.from_list(f"{name}_cmap", colors, N=n_colors)

def show_colorblind_palettes():
    """Display all available colorblind-friendly palettes."""
    fig, axes = plt.subplots(len(COLORBLIND_PALETTES), 1, figsize=(10, 2*len(COLORBLIND_PALETTES)))
    
    for i, (name, colors) in enumerate(COLORBLIND_PALETTES.items()):
        ax = axes[i] if len(COLORBLIND_PALETTES) > 1 else axes
        
        # Create color swatches
        for j, color in enumerate(colors):
            ax.add_patch(plt.Rectangle((j, 0), 1, 1, facecolor=color))
        
        ax.set_xlim(0, len(colors))
        ax.set_ylim(0, 1)
        ax.set_title(f'{name} ({len(colors)} colors)')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add hex codes as text
        for j, color in enumerate(colors):
            ax.text(j+0.5, 0.5, color, ha='center', va='center', 
                   color='white' if _is_dark_color(color) else 'black', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def _is_dark_color(hex_color):
    """Check if a hex color is dark (for text color selection)."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
    return luminance < 0.5