// Pooling operations: MaxPool, AvgPool, AdaptivePool

struct PoolParams {
    batchSize: u32,
    channels: u32,
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
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: PoolParams;

// Max pooling
@compute @workgroup_size(16, 16, 1)
fn maxPool2d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch = global_id.z;
    let outY = global_id.y;
    let outX = global_id.x;

    if (batch >= params.batchSize || outY >= params.outHeight || outX >= params.outWidth) {
        return;
    }

    for (var c = 0u; c < params.channels; c = c + 1u) {
        var maxVal = -3.40282347e+38; // -FLT_MAX

        for (var kh = 0u; kh < params.kernelHeight; kh = kh + 1u) {
            for (var kw = 0u; kw < params.kernelWidth; kw = kw + 1u) {
                let inY = i32(outY * params.strideHeight) + i32(kh) - i32(params.padHeight);
                let inX = i32(outX * params.strideWidth) + i32(kw) - i32(params.padWidth);

                if (inY >= 0 && inY < i32(params.inHeight) && inX >= 0 && inX < i32(params.inWidth)) {
                    let inputIdx = batch * params.channels * params.inHeight * params.inWidth +
                                 c * params.inHeight * params.inWidth +
                                 u32(inY) * params.inWidth +
                                 u32(inX);

                    maxVal = max(maxVal, input[inputIdx]);
                }
            }
        }

        let outputIdx = batch * params.channels * params.outHeight * params.outWidth +
                       c * params.outHeight * params.outWidth +
                       outY * params.outWidth +
                       outX;

        output[outputIdx] = maxVal;
    }
}

// Average pooling
@compute @workgroup_size(16, 16, 1)
fn avgPool2d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch = global_id.z;
    let outY = global_id.y;
    let outX = global_id.x;

    if (batch >= params.batchSize || outY >= params.outHeight || outX >= params.outWidth) {
        return;
    }

    for (var c = 0u; c < params.channels; c = c + 1u) {
        var sum = 0.0;
        var count = 0u;

        for (var kh = 0u; kh < params.kernelHeight; kh = kh + 1u) {
            for (var kw = 0u; kw < params.kernelWidth; kw = kw + 1u) {
                let inY = i32(outY * params.strideHeight) + i32(kh) - i32(params.padHeight);
                let inX = i32(outX * params.strideWidth) + i32(kw) - i32(params.padWidth);

                if (inY >= 0 && inY < i32(params.inHeight) && inX >= 0 && inX < i32(params.inWidth)) {
                    let inputIdx = batch * params.channels * params.inHeight * params.inWidth +
                                 c * params.inHeight * params.inWidth +
                                 u32(inY) * params.inWidth +
                                 u32(inX);

                    sum = sum + input[inputIdx];
                    count = count + 1u;
                }
            }
        }

        let outputIdx = batch * params.channels * params.outHeight * params.outWidth +
                       c * params.outHeight * params.outWidth +
                       outY * params.outWidth +
                       outX;

        output[outputIdx] = sum / f32(count);
    }
}

// Global average pooling
@group(0) @binding(0) var<storage, read> globalInput: array<f32>;
@group(0) @binding(1) var<storage, read_write> globalOutput: array<f32>;

struct GlobalPoolParams {
    batchSize: u32,
    channels: u32,
    height: u32,
    width: u32,
}

@group(0) @binding(2) var<uniform> globalParams: GlobalPoolParams;

@compute @workgroup_size(256, 1, 1)
fn globalAvgPool(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let batch = idx / globalParams.channels;
    let channel = idx % globalParams.channels;

    if (batch >= globalParams.batchSize || channel >= globalParams.channels) {
        return;
    }

    var sum = 0.0;

    let spatialSize = globalParams.height * globalParams.width;
    let baseIdx = batch * globalParams.channels * spatialSize + channel * spatialSize;

    for (var i = 0u; i < spatialSize; i = i + 1u) {
        sum = sum + globalInput[baseIdx + i];
    }

    globalOutput[idx] = sum / f32(spatialSize);
}

// Global max pooling
@compute @workgroup_size(256, 1, 1)
fn globalMaxPool(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let batch = idx / globalParams.channels;
    let channel = idx % globalParams.channels;

    if (batch >= globalParams.batchSize || channel >= globalParams.channels) {
        return;
    }

    var maxVal = -3.40282347e+38;

    let spatialSize = globalParams.height * globalParams.width;
    let baseIdx = batch * globalParams.channels * spatialSize + channel * spatialSize;

    for (var i = 0u; i < spatialSize; i = i + 1u) {
        maxVal = max(maxVal, globalInput[baseIdx + i]);
    }

    globalOutput[idx] = maxVal;
}

// Adaptive pooling - automatically calculates stride/kernel
struct AdaptivePoolParams {
    batchSize: u32,
    channels: u32,
    inHeight: u32,
    inWidth: u32,
    outHeight: u32,
    outWidth: u32,
}

@group(0) @binding(0) var<storage, read> adaptiveInput: array<f32>;
@group(0) @binding(1) var<storage, read_write> adaptiveOutput: array<f32>;
@group(0) @binding(2) var<uniform> adaptiveParams: AdaptivePoolParams;

fn startIndex(idx: u32, outSize: u32, inSize: u32) -> u32 {
    return (idx * inSize) / outSize;
}

fn endIndex(idx: u32, outSize: u32, inSize: u32) -> u32 {
    return ((idx + 1u) * inSize + outSize - 1u) / outSize;
}

@compute @workgroup_size(16, 16, 1)
fn adaptiveAvgPool2d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch = global_id.z;
    let outY = global_id.y;
    let outX = global_id.x;

    if (batch >= adaptiveParams.batchSize ||
        outY >= adaptiveParams.outHeight ||
        outX >= adaptiveParams.outWidth) {
        return;
    }

    let startY = startIndex(outY, adaptiveParams.outHeight, adaptiveParams.inHeight);
    let endY = endIndex(outY, adaptiveParams.outHeight, adaptiveParams.inHeight);
    let startX = startIndex(outX, adaptiveParams.outWidth, adaptiveParams.inWidth);
    let endX = endIndex(outX, adaptiveParams.outWidth, adaptiveParams.inWidth);

    for (var c = 0u; c < adaptiveParams.channels; c = c + 1u) {
        var sum = 0.0;
        var count = 0u;

        for (var y = startY; y < endY; y = y + 1u) {
            for (var x = startX; x < endX; x = x + 1u) {
                let inputIdx = batch * adaptiveParams.channels * adaptiveParams.inHeight * adaptiveParams.inWidth +
                             c * adaptiveParams.inHeight * adaptiveParams.inWidth +
                             y * adaptiveParams.inWidth +
                             x;

                sum = sum + adaptiveInput[inputIdx];
                count = count + 1u;
            }
        }

        let outputIdx = batch * adaptiveParams.channels * adaptiveParams.outHeight * adaptiveParams.outWidth +
                       c * adaptiveParams.outHeight * adaptiveParams.outWidth +
                       outY * adaptiveParams.outWidth +
                       outX;

        adaptiveOutput[outputIdx] = sum / f32(count);
    }
}

@compute @workgroup_size(16, 16, 1)
fn adaptiveMaxPool2d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch = global_id.z;
    let outY = global_id.y;
    let outX = global_id.x;

    if (batch >= adaptiveParams.batchSize ||
        outY >= adaptiveParams.outHeight ||
        outX >= adaptiveParams.outWidth) {
        return;
    }

    let startY = startIndex(outY, adaptiveParams.outHeight, adaptiveParams.inHeight);
    let endY = endIndex(outY, adaptiveParams.outHeight, adaptiveParams.inHeight);
    let startX = startIndex(outX, adaptiveParams.outWidth, adaptiveParams.inWidth);
    let endX = endIndex(outX, adaptiveParams.outWidth, adaptiveParams.inWidth);

    for (var c = 0u; c < adaptiveParams.channels; c = c + 1u) {
        var maxVal = -3.40282347e+38;

        for (var y = startY; y < endY; y = y + 1u) {
            for (var x = startX; x < endX; x = x + 1u) {
                let inputIdx = batch * adaptiveParams.channels * adaptiveParams.inHeight * adaptiveParams.inWidth +
                             c * adaptiveParams.inHeight * adaptiveParams.inWidth +
                             y * adaptiveParams.inWidth +
                             x;

                maxVal = max(maxVal, adaptiveInput[inputIdx]);
            }
        }

        let outputIdx = batch * adaptiveParams.channels * adaptiveParams.outHeight * adaptiveParams.outWidth +
                       c * adaptiveParams.outHeight * adaptiveParams.outWidth +
                       outY * adaptiveParams.outWidth +
                       outX;

        adaptiveOutput[outputIdx] = maxVal;
    }
}
