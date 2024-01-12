def reconstruct(model, calibration_set, reconstruction_method='layer_wise'):
    """
    Reconstruct the quantized model using the calibration set.

    :param model: The quantized model.
    :param calibration_set: The calibration set for reconstruction.
    :param reconstruction_method: The method of reconstruction ('layer_wise', 'block_wise', etc.)
    """
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            print(f"Reconstructing layer: {name}")

            # 재구성 방법에 따라 적절한 재구성 함수 호출
            if reconstruction_method == 'layer_wise':
                layer_reconstruction(model, module, calibration_set)
            # 다른 재구성 방법들에 대한 처리는 여기에 추가
            # 예: elif reconstruction_method == 'block_wise': block_reconstruction(...)
            else:
                raise NotImplementedError(f"Reconstruction method '{reconstruction_method}' not implemented")

    print("Reconstruction completed.")