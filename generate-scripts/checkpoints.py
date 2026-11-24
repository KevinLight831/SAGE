"""Load Diffusers component weights independently of the checkpoint filename."""


def load_component_weights(path, text_encoder, unet=None, component="auto"):
    import torch

    weights = torch.load(path, map_location="cpu", weights_only=True)
    weights = weights.get("state_dict", weights)
    prefixed = {
        key[len("text_encoder."):]: value
        for key, value in weights.items()
        if key.startswith("text_encoder.text_model.")
    }
    if prefixed:
        encoder_weights = prefixed
    else:
        encoder_weights = weights
    if component == "auto":
        component = "text_encoder" if any(
            key.startswith("text_model.") for key in encoder_weights
        ) else "unet"
    if component == "text_encoder":
        expected = text_encoder.state_dict()
        # Transformers versions differ in whether this buffer is persistent.
        key = "text_model.embeddings.position_ids"
        if key in expected:
            encoder_weights.setdefault(key, expected[key])
        else:
            encoder_weights.pop(key, None)
        text_encoder.load_state_dict(encoder_weights, strict=True)
    elif unet is not None:
        unet.load_state_dict(weights, strict=True)
    else:
        raise ValueError("This workflow requires a compatible text-encoder checkpoint")
