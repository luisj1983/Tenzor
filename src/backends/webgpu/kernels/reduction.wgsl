// Reduction operations: sum, mean, max, min, prod, etc.

struct ReductionParams {
    inputSize: u32,
    outputSize: u32,
    reduceSize: u32,
    keepDims: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: ReductionParams;

var<workgroup> sharedData: array<f32, 256>;

// Reduction sum
@compute @workgroup_size(256, 1, 1)
fn reduceSum(@builtin(global_invocation_id) global_id: vec3<u32>,
             @builtin(local_invocation_id) local_id: vec3<u32>) {
    let outIdx = global_id.y;
    let tid = local_id.x;

    if (outIdx >= params.outputSize) {
        return;
    }

    let baseIdx = outIdx * params.reduceSize;

    var sum = 0.0;
    for (var i = tid; i < params.reduceSize; i = i + 256u) {
        sum = sum + input[baseIdx + i];
    }

    sharedData[tid] = sum;
    workgroupBarrier();

    // Reduction in shared memory
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            sharedData[tid] = sharedData[tid] + sharedData[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[outIdx] = sharedData[0];
    }
}

// Reduction mean
@compute @workgroup_size(256, 1, 1)
fn reduceMean(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>) {
    let outIdx = global_id.y;
    let tid = local_id.x;

    if (outIdx >= params.outputSize) {
        return;
    }

    let baseIdx = outIdx * params.reduceSize;

    var sum = 0.0;
    for (var i = tid; i < params.reduceSize; i = i + 256u) {
        sum = sum + input[baseIdx + i];
    }

    sharedData[tid] = sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            sharedData[tid] = sharedData[tid] + sharedData[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[outIdx] = sharedData[0] / f32(params.reduceSize);
    }
}

// Reduction max
@compute @workgroup_size(256, 1, 1)
fn reduceMax(@builtin(global_invocation_id) global_id: vec3<u32>,
             @builtin(local_invocation_id) local_id: vec3<u32>) {
    let outIdx = global_id.y;
    let tid = local_id.x;

    if (outIdx >= params.outputSize) {
        return;
    }

    let baseIdx = outIdx * params.reduceSize;

    var maxVal = -3.40282347e+38;
    for (var i = tid; i < params.reduceSize; i = i + 256u) {
        maxVal = max(maxVal, input[baseIdx + i]);
    }

    sharedData[tid] = maxVal;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            sharedData[tid] = max(sharedData[tid], sharedData[tid + stride]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[outIdx] = sharedData[0];
    }
}

// Reduction min
@compute @workgroup_size(256, 1, 1)
fn reduceMin(@builtin(global_invocation_id) global_id: vec3<u32>,
             @builtin(local_invocation_id) local_id: vec3<u32>) {
    let outIdx = global_id.y;
    let tid = local_id.x;

    if (outIdx >= params.outputSize) {
        return;
    }

    let baseIdx = outIdx * params.reduceSize;

    var minVal = 3.40282347e+38;
    for (var i = tid; i < params.reduceSize; i = i + 256u) {
        minVal = min(minVal, input[baseIdx + i]);
    }

    sharedData[tid] = minVal;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            sharedData[tid] = min(sharedData[tid], sharedData[tid + stride]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[outIdx] = sharedData[0];
    }
}

// Reduction product
@compute @workgroup_size(256, 1, 1)
fn reduceProd(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>) {
    let outIdx = global_id.y;
    let tid = local_id.x;

    if (outIdx >= params.outputSize) {
        return;
    }

    let baseIdx = outIdx * params.reduceSize;

    var prod = 1.0;
    for (var i = tid; i < params.reduceSize; i = i + 256u) {
        prod = prod * input[baseIdx + i];
    }

    sharedData[tid] = prod;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            sharedData[tid] = sharedData[tid] * sharedData[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[outIdx] = sharedData[0];
    }
}

// ArgMax - returns indices
@group(0) @binding(0) var<storage, read> argmaxInput: array<f32>;
@group(0) @binding(1) var<storage, read_write> argmaxOutput: array<u32>;
@group(0) @binding(2) var<uniform> argmaxParams: ReductionParams;

struct ArgMaxData {
    value: f32,
    index: u32,
}

var<workgroup> sharedArgMax: array<ArgMaxData, 256>;

@compute @workgroup_size(256, 1, 1)
fn argMax(@builtin(global_invocation_id) global_id: vec3<u32>,
          @builtin(local_invocation_id) local_id: vec3<u32>) {
    let outIdx = global_id.y;
    let tid = local_id.x;

    if (outIdx >= argmaxParams.outputSize) {
        return;
    }

    let baseIdx = outIdx * argmaxParams.reduceSize;

    var maxData: ArgMaxData;
    maxData.value = -3.40282347e+38;
    maxData.index = 0u;

    for (var i = tid; i < argmaxParams.reduceSize; i = i + 256u) {
        let val = argmaxInput[baseIdx + i];
        if (val > maxData.value) {
            maxData.value = val;
            maxData.index = i;
        }
    }

    sharedArgMax[tid] = maxData;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            if (sharedArgMax[tid + stride].value > sharedArgMax[tid].value) {
                sharedArgMax[tid] = sharedArgMax[tid + stride];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        argmaxOutput[outIdx] = sharedArgMax[0].index;
    }
}

// ArgMin - returns indices
@compute @workgroup_size(256, 1, 1)
fn argMin(@builtin(global_invocation_id) global_id: vec3<u32>,
          @builtin(local_invocation_id) local_id: vec3<u32>) {
    let outIdx = global_id.y;
    let tid = local_id.x;

    if (outIdx >= argmaxParams.outputSize) {
        return;
    }

    let baseIdx = outIdx * argmaxParams.reduceSize;

    var minData: ArgMaxData;
    minData.value = 3.40282347e+38;
    minData.index = 0u;

    for (var i = tid; i < argmaxParams.reduceSize; i = i + 256u) {
        let val = argmaxInput[baseIdx + i];
        if (val < minData.value) {
            minData.value = val;
            minData.index = i;
        }
    }

    sharedArgMax[tid] = minData;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            if (sharedArgMax[tid + stride].value < sharedArgMax[tid].value) {
                sharedArgMax[tid] = sharedArgMax[tid + stride];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        argmaxOutput[outIdx] = sharedArgMax[0].index;
    }
}

// Variance reduction
@compute @workgroup_size(256, 1, 1)
fn reduceVariance(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>) {
    let outIdx = global_id.y;
    let tid = local_id.x;

    if (outIdx >= params.outputSize) {
        return;
    }

    let baseIdx = outIdx * params.reduceSize;

    // First pass: compute mean
    var sum = 0.0;
    for (var i = tid; i < params.reduceSize; i = i + 256u) {
        sum = sum + input[baseIdx + i];
    }

    sharedData[tid] = sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            sharedData[tid] = sharedData[tid] + sharedData[tid + stride];
        }
        workgroupBarrier();
    }

    let mean = sharedData[0] / f32(params.reduceSize);

    // Second pass: compute variance
    var varSum = 0.0;
    for (var i = tid; i < params.reduceSize; i = i + 256u) {
        let diff = input[baseIdx + i] - mean;
        varSum = varSum + diff * diff;
    }

    sharedData[tid] = varSum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            sharedData[tid] = sharedData[tid] + sharedData[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[outIdx] = sharedData[0] / f32(params.reduceSize);
    }
}

// Standard deviation reduction
@compute @workgroup_size(256, 1, 1)
fn reduceStd(@builtin(global_invocation_id) global_id: vec3<u32>,
             @builtin(local_invocation_id) local_id: vec3<u32>) {
    let outIdx = global_id.y;
    let tid = local_id.x;

    if (outIdx >= params.outputSize) {
        return;
    }

    let baseIdx = outIdx * params.reduceSize;

    // First pass: compute mean
    var sum = 0.0;
    for (var i = tid; i < params.reduceSize; i = i + 256u) {
        sum = sum + input[baseIdx + i];
    }

    sharedData[tid] = sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            sharedData[tid] = sharedData[tid] + sharedData[tid + stride];
        }
        workgroupBarrier();
    }

    let mean = sharedData[0] / f32(params.reduceSize);

    // Second pass: compute variance
    var varSum = 0.0;
    for (var i = tid; i < params.reduceSize; i = i + 256u) {
        let diff = input[baseIdx + i] - mean;
        varSum = varSum + diff * diff;
    }

    sharedData[tid] = varSum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            sharedData[tid] = sharedData[tid] + sharedData[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let variance = sharedData[0] / f32(params.reduceSize);
        output[outIdx] = sqrt(variance);
    }
}

// L2 norm reduction
@compute @workgroup_size(256, 1, 1)
fn reduceNorm(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>) {
    let outIdx = global_id.y;
    let tid = local_id.x;

    if (outIdx >= params.outputSize) {
        return;
    }

    let baseIdx = outIdx * params.reduceSize;

    var sumSq = 0.0;
    for (var i = tid; i < params.reduceSize; i = i + 256u) {
        let val = input[baseIdx + i];
        sumSq = sumSq + val * val;
    }

    sharedData[tid] = sumSq;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            sharedData[tid] = sharedData[tid] + sharedData[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[outIdx] = sqrt(sharedData[0]);
    }
}
