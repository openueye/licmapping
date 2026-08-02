#include <tuple>
#include <vector>

#include <torch/extension.h>

#include "rasterizer/rasterizer.h"

namespace {

void check_cuda_float(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_shape(
    const torch::Tensor& tensor,
    std::initializer_list<int64_t> expected,
    const char* name) {
    TORCH_CHECK(tensor.dim() == static_cast<int64_t>(expected.size()), name, " has invalid rank");
    size_t index = 0;
    for (const auto size : expected) {
        TORCH_CHECK(
            size < 0 || tensor.size(static_cast<int64_t>(index)) == size,
            name,
            " has invalid shape");
        ++index;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> rasterize(
    torch::Tensor means3d,
    torch::Tensor means2d,
    torch::Tensor dc,
    torch::Tensor sh,
    torch::Tensor opacities,
    torch::Tensor scales,
    torch::Tensor rotations,
    torch::Tensor background,
    torch::Tensor viewmatrix,
    torch::Tensor projmatrix,
    torch::Tensor campos,
    int64_t image_height,
    int64_t image_width,
    double tanfovx,
    double tanfovy,
    double limx_neg,
    double limx_pos,
    double limy_neg,
    double limy_pos,
    int64_t sh_degree,
    double scale_modifier,
    double lambda_erank,
    bool prefiltered,
    bool debug,
    bool no_color) {
    check_cuda_float(means3d, "means3d");
    check_cuda_float(means2d, "means2d");
    check_cuda_float(dc, "dc");
    check_cuda_float(sh, "sh");
    check_cuda_float(opacities, "opacities");
    check_cuda_float(scales, "scales");
    check_cuda_float(rotations, "rotations");
    check_cuda_float(background, "background");
    check_cuda_float(viewmatrix, "viewmatrix");
    check_cuda_float(projmatrix, "projmatrix");
    check_cuda_float(campos, "campos");

    const auto count = means3d.size(0);
    check_shape(means3d, { -1, 3 }, "means3d");
    TORCH_CHECK(
        count >= 2,
        "LIC reference rasterizer requires at least two Gaussian rows; "
        "its duplicateWithKeys kernel has a one-row edge case");
    check_shape(means2d, { -1, 3 }, "means2d");
    check_shape(dc, { -1, 1, 3 }, "dc");
    check_shape(opacities, { -1, 1 }, "opacities");
    check_shape(scales, { -1, 3 }, "scales");
    check_shape(rotations, { -1, 4 }, "rotations");
    check_shape(background, { 3 }, "background");
    check_shape(viewmatrix, { 4, 4 }, "viewmatrix");
    check_shape(projmatrix, { 4, 4 }, "projmatrix");
    check_shape(campos, { 3 }, "campos");
    TORCH_CHECK(means2d.size(0) == count, "means2d row count must match means3d");
    TORCH_CHECK(dc.size(0) == count, "dc row count must match means3d");
    TORCH_CHECK(opacities.size(0) == count, "opacities row count must match means3d");
    TORCH_CHECK(scales.size(0) == count, "scales row count must match means3d");
    TORCH_CHECK(rotations.size(0) == count, "rotations row count must match means3d");
    TORCH_CHECK(sh_degree >= 0 && sh_degree <= 3, "sh_degree must be in [0, 3]");
    TORCH_CHECK(image_height > 0 && image_width > 0, "image dimensions must be positive");
    TORCH_CHECK(
        sh.numel() == 0 ||
            (sh.dim() == 3 && sh.size(0) == count && sh.size(2) == 3),
        "sh must be empty or have shape [N, M, 3]");
    TORCH_CHECK(
        means3d.device() == background.device() &&
            means3d.device() == means2d.device() &&
            means3d.device() == dc.device() &&
            means3d.device() == sh.device() &&
            means3d.device() == opacities.device() &&
            means3d.device() == scales.device() &&
            means3d.device() == rotations.device() &&
            means3d.device() == viewmatrix.device() &&
            means3d.device() == projmatrix.device() &&
            means3d.device() == campos.device(),
        "all tensors must share a CUDA device");

    // The reference wrapper deliberately receives empty precomputed-color and
    // covariance tensors: colors come from SH and covariance from scale/rotation.
    auto empty = torch::empty({0}, means3d.options());
    GaussianRasterizationSettings settings(
        static_cast<int>(image_height),
        static_cast<int>(image_width),
        static_cast<float>(tanfovx),
        static_cast<float>(tanfovy),
        static_cast<float>(limx_neg),
        static_cast<float>(limx_pos),
        static_cast<float>(limy_neg),
        static_cast<float>(limy_pos),
        background,
        static_cast<float>(scale_modifier),
        viewmatrix,
        projmatrix,
        static_cast<int>(sh_degree),
        campos,
        prefiltered,
        debug,
        no_color,
        static_cast<float>(lambda_erank));
    GaussianRasterizer rasterizer(settings);
    return rasterizer.forward(
        means3d,
        means2d,
        opacities,
        dc,
        sh,
        empty,
        scales,
        rotations,
        empty);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "rasterize",
        &rasterize,
        "Differentiable Gaussian-LIC rasterization");
}
