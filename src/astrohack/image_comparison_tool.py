from astrohack.core.image_compare_tool import image_comparison_chunk


def compare_fits_image(
        image: str,
        reference_image: str,
        telescope_name: str,
        destination: str,
        comparison: str = 'direct',
        plot_data: bool = False,
        plot_percentuals: bool = False,
        plot_divided_image: bool = False,
        plot_scatter: bool = True,
        export_to_fits: bool = False,
        colormap: str = 'viridis',
        dpi: int = 300,
        display: bool = False
):
    param_dict = locals()

    param_dict['this_image'] = image
    param_dict['this_reference_image'] = reference_image

    image_comparison_chunk(param_dict)
