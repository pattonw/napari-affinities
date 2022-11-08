import numpy as np


def get_data(image):
    image = image.data[0] if image.multiscale else image.data
    if not all(
        hasattr(image, attr) for attr in ("shape", "ndim", "__getitem__")
    ):
        image = np.asanyarray(image)
    return image


def corner_pixels_multiscale(layer):
    # layer.corner_pixels are with respect to the currently used resolution level (layer.data_level)
    # -> convert to reference highest resolution level (layer.data[0]), which is used by stardist
    factor = layer.downsample_factors[layer.data_level]
    scaled_corner = np.round(layer.corner_pixels * factor).astype(int)
    shape_max = layer.data[0].shape
    # if layer.rgb -> len(shape_max) == 1 + len(factor)
    for i in range(len(factor)):
        scaled_corner[:, i] = np.clip(scaled_corner[:, i], 0, shape_max[i])
    return scaled_corner


def get_fov_data(viewer, model, image, fov_image:bool, axes):
    offsets = model.config["mws"]["offsets"]
    ndim = len(offsets[0])
    x = get_data(image)
    if fov_image and ndim == 2 and viewer.dims.ndisplay == 2:
        # it's all a big mess based on shaky assumptions...
        if viewer is None:
            raise RuntimeError("viewer is None")
        if image.rgb and axes[-1] != "c":
            raise RuntimeError(
                "rgb image must have channels as last axis/dimension"
            )

        def get_slice_not_displayed(vdim, idim):
            # vdim: dimension index wrt. viewer
            # idim: dimension index wrt. image
            if axes[idim] == "T":
                # if timelapse, return visible/selected frame
                return slice(
                    viewer.dims.current_step[vdim],
                    1 + viewer.dims.current_step[vdim],
                )
            elif axes[idim] == "c":
                # if channel, return entire dimension
                return slice(0, x.shape[idim])
            else:
                return None

        corner_pixels = (
            corner_pixels_multiscale(image)
            if image.multiscale
            else image.corner_pixels
        )
        n_corners = corner_pixels.shape[1]
        assert n_corners <= x.ndim

        # map viewer dimension index to image dimension index
        n_dims = x.ndim - (1 if image.rgb else 0)
        viewer_dim_to_image_dim = dict(
            zip(np.arange(viewer.dims.ndim)[-n_dims:], range(n_dims))
        )
        # map viewer dimension index to corner pixel
        viewer_dim_to_corner = dict(
            zip(
                np.arange(viewer.dims.ndim)[-n_corners:],
                zip(corner_pixels[0], corner_pixels[1]),
            )
        )
        print(viewer_dim_to_image_dim)
        sl = [None] * x.ndim
        for vdim in range(viewer.dims.ndim):
            idim = viewer_dim_to_image_dim.get(vdim)
            c = viewer_dim_to_corner.get(vdim)
            print(vdim, idim, c)
            if c is not None:
                if vdim in viewer.dims.displayed:
                    fr, to = c
                    sl[idim] = None if fr == to else slice(fr, to)
                else:
                    sl[idim] = get_slice_not_displayed(vdim, idim)
            else:
                # assert vdim in viewer.dims.not_displayed
                if idim is not None:
                    sl[idim] = get_slice_not_displayed(vdim, idim)

        if image.rgb:
            idim = x.ndim - 1
            # set channel slice here, since channel of rgb image not part of viewer dimensions
            assert sl[idim] is None and axes[idim] == "c"
            sl[idim] = get_slice_not_displayed(None, idim)

        sl = tuple(sl)

        print(sl, axes)

        invalid_axes = "".join(a for s, a in zip(sl, axes) if s is None)
        if len(invalid_axes) > 0:
            raise ValueError(
                f"Invalid field of view range for axes {invalid_axes}"
            )
        return sl
    else:
        return None
