from collections import defaultdict

def get_module_by_name_ViT_Simple(model, module_name):
    parts = module_name.split('.')
    for part in parts:
        if part.isdigit():
            model = model[int(part)]
        else:
            model = getattr(model, part)
    return model


def get_axis_to_perm_ViT_Simple(model):
    axis_to_perm = defaultdict(list)
    for i in range(len(model.transformer.layers)):
        axis_to_perm[f"group_{i}"] = [
            f"transformer.layers.{i}.1.net.1",
            f"transformer.layers.{i}.1.net.3",
        ]
    return axis_to_perm

