from magicgui.widgets import create_widget, FunctionGui


def layer_choice_widget(viewer, annotation, **kwargs) -> FunctionGui:
    widget = create_widget(annotation=annotation, **kwargs)
    widget.reset_choices()
    viewer.layers.events.inserted.connect(widget.reset_choices)
    return widget
