// Activation functions for neural networks

struct ActivationParams {
    size: u32,
    alpha: f32,  // For LeakyReLU, ELU, etc.
    beta: f32,   // For additional parameters
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: ActivationParams;

// ReLU activation
@compute @workgroup_size(256, 1, 1)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = max(0.0, input[idx]);
}

// Leaky ReLU
@compute @workgroup_size(256, 1, 1)
fn leakyRelu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    let x = input[idx];
    output[idx] = select(params.alpha * x, x, x > 0.0);
}

// ELU (Exponential Linear Unit)
@compute @workgroup_size(256, 1, 1)
fn elu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    let x = input[idx];
    output[idx] = select(params.alpha * (exp(x) - 1.0), x, x > 0.0);
}

// GELU (Gaussian Error Linear Unit)
@compute @workgroup_size(256, 1, 1)
fn gelu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    let x = input[idx];
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let sqrt2OverPi = 0.7978845608;
    let inner = sqrt2OverPi * (x + 0.044715 * x * x * x);
    output[idx] = 0.5 * x * (1.0 + tanh(inner));
}

// Sigmoid
@compute @workgroup_size(256, 1, 1)
fn sigmoid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = 1.0 / (1.0 + exp(-input[idx]));
}

// Tanh
@compute @workgroup_size(256, 1, 1)
fn tanhActivation(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = tanh(input[idx]);
}

// Softplus: log(1 + exp(x))
@compute @workgroup_size(256, 1, 1)
fn softplus(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    let x = input[idx];
    // Numerically stable version
    output[idx] = select(log(1.0 + exp(x)), x, x > 20.0);
}

// Swish/SiLU: x * sigmoid(x)
@compute @workgroup_size(256, 1, 1)
fn swish(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    let x = input[idx];
    output[idx] = x / (1.0 + exp(-x));
}

// Mish: x * tanh(softplus(x))
@compute @workgroup_size(256, 1, 1)
fn mish(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    let x = input[idx];
    let sp = select(log(1.0 + exp(x)), x, x > 20.0);
    output[idx] = x * tanh(sp);
}

// Hardswish: x * ReLU6(x + 3) / 6
@compute @workgroup_size(256, 1, 1)
fn hardswish(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    let x = input[idx];
    output[idx] = x * min(max(x + 3.0, 0.0), 6.0) / 6.0;
}

// Hardsigmoid: ReLU6(x + 3) / 6
@compute @workgroup_size(256, 1, 1)
fn hardsigmoid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = min(max(input[idx] + 3.0, 0.0), 6.0) / 6.0;
}

// Softmax - Stage 1: Find max
var<workgroup> sharedMax: array<f32, 256>;

@group(0) @binding(0) var<storage, read> softmaxInput: array<f32>;
@group(0) @binding(1) var<storage, read_write> softmaxOutput: array<f32>;
@group(0) @binding(2) var<storage, read_write> maxValues: array<f32>;

struct SoftmaxParams {
    batchSize: u32,
    dim: u32,
}

@group(0) @binding(3) var<uniform> softmaxParams: SoftmaxParams;

@compute @workgroup_size(256, 1, 1)
fn softmaxMax(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>) {
    let batch = global_id.y;
    let tid = local_id.x;

    if (batch >= softmaxParams.batchSize) {
        return;
    }

    let baseIdx = batch * softmaxParams.dim;

    var maxVal = -3.40282347e+38;
    for (var i = tid; i < softmaxParams.dim; i = i + 256u) {
        maxVal = max(maxVal, softmaxInput[baseIdx + i]);
    }

    sharedMax[tid] = maxVal;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            sharedMax[tid] = max(sharedMax[tid], sharedMax[tid + stride]);
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        maxValues[batch] = sharedMax[0];
    }
}

// Softmax - Stage 2: Compute exp and sum
var<workgroup> sharedSum: array<f32, 256>;

@group(0) @binding(4) var<storage, read_write> sumValues: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn softmaxExpSum(@builtin(global_invocation_id) global_id: vec3<u32>,
                 @builtin(local_invocation_id) local_id: vec3<u32>) {
    let batch = global_id.y;
    let tid = local_id.x;

    if (batch >= softmaxParams.batchSize) {
        return;
    }

    let baseIdx = batch * softmaxParams.dim;
    let maxVal = maxValues[batch];

    var sum = 0.0;
    for (var i = tid; i < softmaxParams.dim; i = i + 256u) {
        let expVal = exp(softmaxInput[baseIdx + i] - maxVal);
        softmaxOutput[baseIdx + i] = expVal;
        sum = sum + expVal;
    }

    sharedSum[tid] = sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            sharedSum[tid] = sharedSum[tid] + sharedSum[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        sumValues[batch] = sharedSum[0];
    }
}

// Softmax - Stage 3: Normalize
@compute @workgroup_size(256, 1, 1)
fn softmaxNormalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch = global_id.y;
    let idx = global_id.x;

    if (batch >= softmaxParams.batchSize || idx >= softmaxParams.dim) {
        return;
    }

    let globalIdx = batch * softmaxParams.dim + idx;
    let sum = sumValues[batch];

    softmaxOutput[globalIdx] = softmaxOutput[globalIdx] / sum;
}

// Log Softmax
@compute @workgroup_size(256, 1, 1)
fn logSoftmaxNormalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch = global_id.y;
    let idx = global_id.x;

    if (batch >= softmaxParams.batchSize || idx >= softmaxParams.dim) {
        return;
    }

    let globalIdx = batch * softmaxParams.dim + idx;
    let maxVal = maxValues[batch];
    let sum = sumValues[batch];

    softmaxOutput[globalIdx] = softmaxInput[globalIdx] - maxVal - log(sum);
}

// PReLU (Parametric ReLU)
@group(0) @binding(0) var<storage, read> preluInput: array<f32>;
@group(0) @binding(1) var<storage, read> preluWeights: array<f32>;  // One per channel
@group(0) @binding(2) var<storage, read_write> preluOutput: array<f32>;

struct PReLUParams {
    batchSize: u32,
    channels: u32,
    spatialSize: u32,
}

@group(0) @binding(3) var<uniform> preluParams: PReLUParams;

@compute @workgroup_size(256, 1, 1)
fn prelu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    let totalSize = preluParams.batchSize * preluParams.channels * preluParams.spatialSize;
    if (idx >= totalSize) {
        return;
    }

    let channel = (idx / preluParams.spatialSize) % preluParams.channels;
    let x = preluInput[idx];

    preluOutput[idx] = select(preluWeights[channel] * x, x, x > 0.0);
}
