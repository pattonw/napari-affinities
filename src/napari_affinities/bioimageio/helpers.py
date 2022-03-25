from bioimageio.core.resource_io.nodes import Model, ImportedSource

from marshmallow import missing
import torch


def get_torch_module(model: Model) -> torch.nn.Module:
    """
    Get python torch model from a bioimage.io `Model` class
    copied from:
    https://github.com/bioimage-io/core-bioimage-io-python/blob/3364875eec581b5cd5950441915aa00219bbaf18/
    bioimageio/core/prediction_pipeline/_model_adapters/_pytorch_model_adapter.py#L54
    """
    # TODO: This is torch specific. Bioimage-io models support many more
    # model frameworks. How to handle non-torch models still needs to be
    # handled
    # Most notebooks/code I could find related to loading a bioimage-io model
    # worked under the assumption that you knew what model, and thus what
    # framework you would be using

    weight_spec = model.weights.get("pytorch_state_dict")
    assert weight_spec is not None
    assert isinstance(weight_spec.architecture, ImportedSource)
    model_kwargs = weight_spec.kwargs
    joined_kwargs = {} if model_kwargs is missing else dict(model_kwargs)
    model = weight_spec.architecture(**joined_kwargs)

    if weight_spec.source is not None:
        state = torch.load(weight_spec.source)
        model.load_state_dict(state)
    return model


def update_weights(model, weights):
    """
    Package up a trained/finetuned model as a new bioimageio model
    """
    from bioimageio.core.build_spec import build_model

    # create a subfolder to store the files for the new model
    model_root = Path("./sample_data")
    model_root.mkdir(exist_ok=True)

    # create the expected output tensor
    new_output = None
    new_output_path = f"{model_root}/test_output.npy"
    np.save(new_output_path, new_output)

    # add thresholding as post-processing procedure to our model
    preprocessing = [
        [{"name": prep.name, "kwargs": prep.kwargs} for prep in inp.preprocessing]
        for inp in model_resource.inputs
    ]
    postprocessing = [
        [{"name": prep.name, "kwargs": prep.kwargs} for prep in inp.postprocessing]
        for inp in model_resource.outputs
    ]

    # get the model architecture
    # note that this is only necessary for pytorch state dict models
    model_source = get_architecture_source(rdf_doi)

    # we use the `parent` field to indicate that the new model is created based on
    # the nucleus segmentation model we have obtained from bioimage.io
    # this field is optional and only needs to be given for models that are created based on other models from bioimage.io
    # the parent is specified via it's doi and the hash of its rdf file
    model_root_folder = os.path.split(
        model_resource.weights["pytorch_state_dict"].source
    )[0]
    rdf_file = model_root_folder / "rdf.yaml"
    with rdf_file.open("rb") as f:
        rdf_hash = hashlib.sha256(f.read()).hexdigest()
    parent = {"uri": rdf_doi, "sha256": rdf_hash}

    # the name of the new model and where to save the zipped model package
    name = f"{old_model_name}_finetuned"
    zip_path = model_root / f"{name}.zip"

    # `build_model` needs some additional information about the model, like citation information
    # all this additional information is passed as plain python types and will be converted into the bioimageio representation internally
    # for more informantion, check out the function signature
    # https://github.com/bioimage-io/core-bioimage-io-python/blob/main/bioimageio/core/build_spec/build_model.py#L252
    cite = [
        {"text": cite_entry.text, "url": cite_entry.url}
        for cite_entry in model_resource.cite
    ]

    # TODO: provide this option if data being looked at is available on bioimage.io
    # the training data used for the model can also be specified by linking to a dataset available on bioimage.io
    # training_data = {"id": "ilastik/stradist_dsb_training_data"}

    # the axes descriptions for the inputs / outputs
    input_axes = model_resource.input_axes
    output_axes = model_resource.output_axes

    # the pytorch_state_dict weight file
    weight_file = model_resource.weights["pytorch_state_dict"].source

    # the path to save the new model with torchscript weights
    zip_path = f"{model_root}/new_model2.zip"

    # build the model! it will be saved to 'zip_path'
    new_model_raw = build_model(
        weight_uri=weight_file,
        test_inputs=model_resource.test_inputs,
        test_outputs=[new_output_path],
        input_axes=input_axes,
        output_axes=output_axes,
        output_path=zip_path,
        name=name,
        description=f"{model_resource.description} (Finetuned with Napari-affinities plugin)",
        authors=[{"name": "Jane Doe"}],  # TODO: let users plug in their own name
        license="CC-BY-4.0",  # TODO: configurable?
        documentation=model_resource.documentation,
        covers=[str(cover) for cover in model_resource.covers],
        tags=model_resource.tags + ["Napari-affinities"],
        cite=cite,
        parent=parent,
        architecture=model_source,
        model_kwargs=model_resource.weights["pytorch_state_dict"].kwargs,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        training_data=model_resource.training_data,  # TODO: add our data here, identify it as finetuning data?
    )
