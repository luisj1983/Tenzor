// Batch Normalization and Layer Normalization

struct BatchNormParams {
    batchSize: u32,
    channels: u32,
    height: u32,
    width: u32,
    epsilon: f32,
    momentum: f32,
    training: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;  // gamma
@group(0) @binding(2) var<storage, read> bias: array<f32>;    // beta
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read_write> runningMean: array<f32>;
@group(0) @binding(5) var<storage, read_write> runningVar: array<f32>;
@group(0) @binding(6) var<uniform> params: BatchNormParams;

// Stage 1: Compute mean per channel
var<workgroup> sharedSum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn batchnormMean(@builtin(global_invocation_id) global_id: vec3<u32>,
                 @builtin(local_invocation_id) local_id: vec3<u32>) {
    let channel = global_id.y;
    let tid = local_id.x;

    if (channel >= params.channels) {
        return;
    }

    let spatialSize = params.height * params.width;
    let totalSize = params.batchSize * spatialSize;

    // Each thread accumulates a subset
    var localSum = 0.0;
    for (var i = tid; i < totalSize; i = i + 256u) {
        let batch = i / spatialSize;
        let spatial = i % spatialSize;
        let idx = batch * params.channels * spatialSize + channel * spatialSize + spatial;
        localSum = localSum + input[idx];
    }

    sharedSum[tid] = localSum;
    workgroupBarrier();

    // Reduction in shared memory
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            sharedSum[tid] = sharedSum[tid] + sharedSum[tid + stride];
        }
        workgroupBarrier();
    }

    // Write result
    if (tid == 0u) {
        runningMean[channel] = sharedSum[0] / f32(totalSize);
    }
}

// Stage 2: Compute variance per channel
@compute @workgroup_size(256, 1, 1)
fn batchnormVariance(@builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(local_invocation_id) local_id: vec3<u32>) {
    let channel = global_id.y;
    let tid = local_id.x;

    if (channel >= params.channels) {
        return;
    }

    let spatialSize = params.height * params.width;
    let totalSize = params.batchSize * spatialSize;
    let mean = runningMean[channel];

    var localSum = 0.0;
    for (var i = tid; i < totalSize; i = i + 256u) {
        let batch = i / spatialSize;
        let spatial = i % spatialSize;
        let idx = batch * params.channels * spatialSize + channel * spatialSize + spatial;
        let diff = input[idx] - mean;
        localSum = localSum + diff * diff;
    }

    sharedSum[tid] = localSum;
    workgroupBarrier();

    // Reduction
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            sharedSum[tid] = sharedSum[tid] + sharedSum[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        runningVar[channel] = sharedSum[0] / f32(totalSize);
    }
}

// Stage 3: Normalize and scale
@compute @workgroup_size(16, 16, 1)
fn batchnormNormalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch = global_id.z;
    let channel = global_id.y;
    let spatial = global_id.x;

    let spatialSize = params.height * params.width;

    if (batch >= params.batchSize || channel >= params.channels || spatial >= spatialSize) {
        return;
    }

    let idx = batch * params.channels * spatialSize + channel * spatialSize + spatial;

    let mean = runningMean[channel];
    let variance = runningVar[channel];
    let invStd = 1.0 / sqrt(variance + params.epsilon);

    let normalized = (input[idx] - mean) * invStd;
    output[idx] = weight[channel] * normalized + bias[channel];
}

// Combined batch normalization (all stages in one)
@compute @workgroup_size(256, 1, 1)
fn batchnormForward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    let spatialSize = params.height * params.width;
    let totalSize = params.batchSize * params.channels * spatialSize;

    if (idx >= totalSize) {
        return;
    }

    let batch = idx / (params.channels * spatialSize);
    let remaining = idx % (params.channels * spatialSize);
    let channel = remaining / spatialSize;
    let spatial = remaining % spatialSize;

    let mean = runningMean[channel];
    let variance = runningVar[channel];
    let invStd = 1.0 / sqrt(variance + params.epsilon);

    let normalized = (input[idx] - mean) * invStd;
    output[idx] = weight[channel] * normalized + bias[channel];
}

// Layer Normalization
struct LayerNormParams {
    batchSize: u32,
    normalizedShape: u32,
    epsilon: f32,
}

@group(0) @binding(0) var<storage, read> lnInput: array<f32>;
@group(0) @binding(1) var<storage, read> lnWeight: array<f32>;
@group(0) @binding(2) var<storage, read> lnBias: array<f32>;
@group(0) @binding(3) var<storage, read_write> lnOutput: array<f32>;
@group(0) @binding(4) var<uniform> lnParams: LayerNormParams;

var<workgroup> lnSharedData: array<f32, 512>;

@compute @workgroup_size(256, 1, 1)
fn layernorm(@builtin(global_invocation_id) global_id: vec3<u32>,
             @builtin(local_invocation_id) local_id: vec3<u32>) {
    let batch = global_id.y;
    let tid = local_id.x;

    if (batch >= lnParams.batchSize) {
        return;
    }

    let baseIdx = batch * lnParams.normalizedShape;

    // Compute mean
    var sum = 0.0;
    for (var i = tid; i < lnParams.normalizedShape; i = i + 256u) {
        sum = sum + lnInput[baseIdx + i];
    }

    lnSharedData[tid] = sum;
    workgroupBarrier();

    // Reduce to get mean
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            lnSharedData[tid] = lnSharedData[tid] + lnSharedData[tid + stride];
        }
        workgroupBarrier();
    }

    let mean = lnSharedData[0] / f32(lnParams.normalizedShape);

    // Compute variance
    var varSum = 0.0;
    for (var i = tid; i < lnParams.normalizedShape; i = i + 256u) {
        let diff = lnInput[baseIdx + i] - mean;
        varSum = varSum + diff * diff;
    }

    lnSharedData[tid] = varSum;
    workgroupBarrier();

    // Reduce to get variance
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            lnSharedData[tid] = lnSharedData[tid] + lnSharedData[tid + stride];
        }
        workgroupBarrier();
    }

    let variance = lnSharedData[0] / f32(lnParams.normalizedShape);
    let invStd = 1.0 / sqrt(variance + lnParams.epsilon);

    // Normalize and scale
    for (var i = tid; i < lnParams.normalizedShape; i = i + 256u) {
        let normalized = (lnInput[baseIdx + i] - mean) * invStd;
        lnOutput[baseIdx + i] = lnWeight[i] * normalized + lnBias[i];
    }
}

// Group Normalization
struct GroupNormParams {
    batchSize: u32,
    channels: u32,
    groups: u32,
    height: u32,
    width: u32,
    epsilon: f32,
}

@group(0) @binding(0) var<storage, read> gnInput: array<f32>;
@group(0) @binding(1) var<storage, read> gnWeight: array<f32>;
@group(0) @binding(2) var<storage, read> gnBias: array<f32>;
@group(0) @binding(3) var<storage, read_write> gnOutput: array<f32>;
@group(0) @binding(4) var<uniform> gnParams: GroupNormParams;

var<workgroup> gnSharedData: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn groupnorm(@builtin(global_invocation_id) global_id: vec3<u32>,
             @builtin(local_invocation_id) local_id: vec3<u32>) {
    let batch = global_id.y;
    let group = global_id.z;
    let tid = local_id.x;

    if (batch >= gnParams.batchSize || group >= gnParams.groups) {
        return;
    }

    let channelsPerGroup = gnParams.channels / gnParams.groups;
    let spatialSize = gnParams.height * gnParams.width;
    let groupSize = channelsPerGroup * spatialSize;

    let baseChannel = group * channelsPerGroup;
    let baseIdx = batch * gnParams.channels * spatialSize + baseChannel * spatialSize;

    // Compute mean
    var sum = 0.0;
    for (var i = tid; i < groupSize; i = i + 256u) {
        sum = sum + gnInput[baseIdx + i];
    }

    gnSharedData[tid] = sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            gnSharedData[tid] = gnSharedData[tid] + gnSharedData[tid + stride];
        }
        workgroupBarrier();
    }

    let mean = gnSharedData[0] / f32(groupSize);

    // Compute variance
    var varSum = 0.0;
    for (var i = tid; i < groupSize; i = i + 256u) {
        let diff = gnInput[baseIdx + i] - mean;
        varSum = varSum + diff * diff;
    }

    gnSharedData[tid] = varSum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            gnSharedData[tid] = gnSharedData[tid] + gnSharedData[tid + stride];
        }
        workgroupBarrier();
    }

    let variance = gnSharedData[0] / f32(groupSize);
    let invStd = 1.0 / sqrt(variance + gnParams.epsilon);

    // Normalize
    for (var i = tid; i < groupSize; i = i + 256u) {
        let channel = baseChannel + (i / spatialSize);
        let normalized = (gnInput[baseIdx + i] - mean) * invStd;
        gnOutput[baseIdx + i] = gnWeight[channel] * normalized + gnBias[channel];
    }
}
