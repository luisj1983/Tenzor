// 2D Convolution shader with multiple optimization strategies

struct Conv2DParams {
    batchSize: u32,
    inChannels: u32,
    outChannels: u32,
    inHeight: u32,
    inWidth: u32,
    outHeight: u32,
    outWidth: u32,
    kernelHeight: u32,
    kernelWidth: u32,
    strideHeight: u32,
    strideWidth: u32,
    padHeight: u32,
    padWidth: u32,
    dilationHeight: u32,
    dilationWidth: u32,
    groups: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Conv2DParams;

// Direct convolution
@compute @workgroup_size(16, 16, 1)
fn conv2d_direct(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch = global_id.z;
    let outY = global_id.y;
    let outX = global_id.x;

    if (batch >= params.batchSize || outY >= params.outHeight || outX >= params.outWidth) {
        return;
    }

    let inChannelsPerGroup = params.inChannels / params.groups;
    let outChannelsPerGroup = params.outChannels / params.groups;

    for (var oc = 0u; oc < params.outChannels; oc = oc + 1u) {
        var sum = 0.0;

        let group = oc / outChannelsPerGroup;
        let inChannelStart = group * inChannelsPerGroup;
        let inChannelEnd = inChannelStart + inChannelsPerGroup;

        for (var ic = inChannelStart; ic < inChannelEnd; ic = ic + 1u) {
            for (var kh = 0u; kh < params.kernelHeight; kh = kh + 1u) {
                for (var kw = 0u; kw < params.kernelWidth; kw = kw + 1u) {
                    let inY = i32(outY * params.strideHeight) + i32(kh * params.dilationHeight) - i32(params.padHeight);
                    let inX = i32(outX * params.strideWidth) + i32(kw * params.dilationWidth) - i32(params.padWidth);

                    if (inY >= 0 && inY < i32(params.inHeight) && inX >= 0 && inX < i32(params.inWidth)) {
                        let inputIdx = batch * params.inChannels * params.inHeight * params.inWidth +
                                     ic * params.inHeight * params.inWidth +
                                     u32(inY) * params.inWidth +
                                     u32(inX);

                        let weightIdx = oc * inChannelsPerGroup * params.kernelHeight * params.kernelWidth +
                                      (ic - inChannelStart) * params.kernelHeight * params.kernelWidth +
                                      kh * params.kernelWidth +
                                      kw;

                        sum = sum + input[inputIdx] * weight[weightIdx];
                    }
                }
            }
        }

        // Add bias
        sum = sum + bias[oc];

        let outputIdx = batch * params.outChannels * params.outHeight * params.outWidth +
                       oc * params.outHeight * params.outWidth +
                       outY * params.outWidth +
                       outX;

        output[outputIdx] = sum;
    }
}

// Optimized convolution with shared memory
const TILE_SIZE: u32 = 16u;
const KERNEL_MAX: u32 = 9u;

var<workgroup> sharedInput: array<f32, 324>; // (TILE_SIZE + 2*KERNEL_MAX) * (TILE_SIZE + 2*KERNEL_MAX)

@compute @workgroup_size(16, 16, 1)
fn conv2d_tiled(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>) {
    let batch = global_id.z;
    let outY = global_id.y;
    let outX = global_id.x;

    if (batch >= params.batchSize || outY >= params.outHeight || outX >= params.outWidth) {
        return;
    }

    // Load input tile into shared memory
    let localX = local_id.x;
    let localY = local_id.y;

    for (var oc = 0u; oc < params.outChannels; oc = oc + 1u) {
        var sum = 0.0;

        for (var ic = 0u; ic < params.inChannels; ic = ic + 1u) {
            // Cooperative loading into shared memory
            for (var i = localY; i < TILE_SIZE + 2u * params.kernelHeight; i = i + TILE_SIZE) {
                for (var j = localX; j < TILE_SIZE + 2u * params.kernelWidth; j = j + TILE_SIZE) {
                    let inY = i32(global_id.y - local_id.y + i) - i32(params.padHeight);
                    let inX = i32(global_id.x - local_id.x + j) - i32(params.padWidth);

                    if (inY >= 0 && inY < i32(params.inHeight) && inX >= 0 && inX < i32(params.inWidth)) {
                        let inputIdx = batch * params.inChannels * params.inHeight * params.inWidth +
                                     ic * params.inHeight * params.inWidth +
                                     u32(inY) * params.inWidth +
                                     u32(inX);
                        sharedInput[i * (TILE_SIZE + 2u * params.kernelWidth) + j] = input[inputIdx];
                    } else {
                        sharedInput[i * (TILE_SIZE + 2u * params.kernelWidth) + j] = 0.0;
                    }
                }
            }

            workgroupBarrier();

            // Compute convolution using shared memory
            for (var kh = 0u; kh < params.kernelHeight; kh = kh + 1u) {
                for (var kw = 0u; kw < params.kernelWidth; kw = kw + 1u) {
                    let sharedY = localY + kh;
                    let sharedX = localX + kw;
                    let sharedIdx = sharedY * (TILE_SIZE + 2u * params.kernelWidth) + sharedX;

                    let weightIdx = oc * params.inChannels * params.kernelHeight * params.kernelWidth +
                                  ic * params.kernelHeight * params.kernelWidth +
                                  kh * params.kernelWidth +
                                  kw;

                    sum = sum + sharedInput[sharedIdx] * weight[weightIdx];
                }
            }

            workgroupBarrier();
        }

        sum = sum + bias[oc];

        let outputIdx = batch * params.outChannels * params.outHeight * params.outWidth +
                       oc * params.outHeight * params.outWidth +
                       outY * params.outWidth +
                       outX;

        output[outputIdx] = sum;
    }
}

// Depthwise convolution (groups == channels)
@compute @workgroup_size(256, 1, 1)
fn conv2d_depthwise(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    let batch = idx / (params.outChannels * params.outHeight * params.outWidth);
    let remaining = idx % (params.outChannels * params.outHeight * params.outWidth);
    let channel = remaining / (params.outHeight * params.outWidth);
    let spatial = remaining % (params.outHeight * params.outWidth);
    let outY = spatial / params.outWidth;
    let outX = spatial % params.outWidth;

    if (batch >= params.batchSize || channel >= params.outChannels ||
        outY >= params.outHeight || outX >= params.outWidth) {
        return;
    }

    var sum = 0.0;

    for (var kh = 0u; kh < params.kernelHeight; kh = kh + 1u) {
        for (var kw = 0u; kw < params.kernelWidth; kw = kw + 1u) {
            let inY = i32(outY * params.strideHeight) + i32(kh) - i32(params.padHeight);
            let inX = i32(outX * params.strideWidth) + i32(kw) - i32(params.padWidth);

            if (inY >= 0 && inY < i32(params.inHeight) && inX >= 0 && inX < i32(params.inWidth)) {
                let inputIdx = batch * params.inChannels * params.inHeight * params.inWidth +
                             channel * params.inHeight * params.inWidth +
                             u32(inY) * params.inWidth +
                             u32(inX);

                let weightIdx = channel * params.kernelHeight * params.kernelWidth +
                              kh * params.kernelWidth +
                              kw;

                sum = sum + input[inputIdx] * weight[weightIdx];
            }
        }
    }

    sum = sum + bias[channel];
    output[idx] = sum;
}
