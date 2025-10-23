// Mathematical operations: elementwise, broadcast, etc.

struct MathParams {
    size: u32,
    alpha: f32,
    beta: f32,
}

@group(0) @binding(0) var<storage, read> inputA: array<f32>;
@group(0) @binding(1) var<storage, read> inputB: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: MathParams;

// Element-wise addition
@compute @workgroup_size(256, 1, 1)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = inputA[idx] + inputB[idx];
}

// Element-wise subtraction
@compute @workgroup_size(256, 1, 1)
fn sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = inputA[idx] - inputB[idx];
}

// Element-wise multiplication
@compute @workgroup_size(256, 1, 1)
fn mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = inputA[idx] * inputB[idx];
}

// Element-wise division
@compute @workgroup_size(256, 1, 1)
fn div(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = inputA[idx] / inputB[idx];
}

// Element-wise power
@compute @workgroup_size(256, 1, 1)
fn pow(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = pow(inputA[idx], inputB[idx]);
}

// Scalar operations
@group(0) @binding(0) var<storage, read> scalarInput: array<f32>;
@group(0) @binding(1) var<storage, read_write> scalarOutput: array<f32>;
@group(0) @binding(2) var<uniform> scalarParams: MathParams;

// Add scalar
@compute @workgroup_size(256, 1, 1)
fn addScalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = scalarInput[idx] + scalarParams.alpha;
}

// Multiply scalar
@compute @workgroup_size(256, 1, 1)
fn mulScalar(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = scalarInput[idx] * scalarParams.alpha;
}

// Scale and add: output = alpha * input + beta
@compute @workgroup_size(256, 1, 1)
fn scaleAdd(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = scalarParams.alpha * scalarInput[idx] + scalarParams.beta;
}

// Unary operations
@compute @workgroup_size(256, 1, 1)
fn sqrt(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = sqrt(scalarInput[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn exp(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = exp(scalarInput[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn log(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = log(scalarInput[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn abs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = abs(scalarInput[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn neg(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = -scalarInput[idx];
}

@compute @workgroup_size(256, 1, 1)
fn sign(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = sign(scalarInput[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn ceil(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = ceil(scalarInput[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn floor(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = floor(scalarInput[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn round(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = round(scalarInput[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn reciprocal(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = 1.0 / scalarInput[idx];
}

@compute @workgroup_size(256, 1, 1)
fn rsqrt(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = 1.0 / sqrt(scalarInput[idx]);
}

// Trigonometric functions
@compute @workgroup_size(256, 1, 1)
fn sin(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = sin(scalarInput[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn cos(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = cos(scalarInput[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn tan(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = tan(scalarInput[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn asin(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = asin(scalarInput[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn acos(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = acos(scalarInput[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn atan(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = atan(scalarInput[idx]);
}

// Comparison operations (return 1.0 or 0.0)
@compute @workgroup_size(256, 1, 1)
fn eq(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = select(0.0, 1.0, inputA[idx] == inputB[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn ne(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = select(0.0, 1.0, inputA[idx] != inputB[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn lt(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = select(0.0, 1.0, inputA[idx] < inputB[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn le(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = select(0.0, 1.0, inputA[idx] <= inputB[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn gt(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = select(0.0, 1.0, inputA[idx] > inputB[idx]);
}

@compute @workgroup_size(256, 1, 1)
fn ge(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = select(0.0, 1.0, inputA[idx] >= inputB[idx]);
}

// Clamp
@compute @workgroup_size(256, 1, 1)
fn clamp(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = clamp(scalarInput[idx], scalarParams.alpha, scalarParams.beta);
}

// Where (ternary select)
@group(0) @binding(0) var<storage, read> condition: array<f32>;
@group(0) @binding(1) var<storage, read> trueVals: array<f32>;
@group(0) @binding(2) var<storage, read> falseVals: array<f32>;
@group(0) @binding(3) var<storage, read_write> whereOutput: array<f32>;

struct WhereParams {
    size: u32,
}

@group(0) @binding(4) var<uniform> whereParams: WhereParams;

@compute @workgroup_size(256, 1, 1)
fn where(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= whereParams.size) {
        return;
    }
    whereOutput[idx] = select(falseVals[idx], trueVals[idx], condition[idx] != 0.0);
}

// Broadcasting operations
struct BroadcastParams {
    outputSize: u32,
    inputASize: u32,
    inputBSize: u32,
    dimA0: u32,
    dimA1: u32,
    dimA2: u32,
    dimA3: u32,
    dimB0: u32,
    dimB1: u32,
    dimB2: u32,
    dimB3: u32,
    dimOut0: u32,
    dimOut1: u32,
    dimOut2: u32,
    dimOut3: u32,
}

@group(0) @binding(0) var<storage, read> broadcastA: array<f32>;
@group(0) @binding(1) var<storage, read> broadcastB: array<f32>;
@group(0) @binding(2) var<storage, read_write> broadcastOutput: array<f32>;
@group(0) @binding(3) var<uniform> broadcastParams: BroadcastParams;

fn computeBroadcastIndex(outIdx: u32, inDim0: u32, inDim1: u32, inDim2: u32, inDim3: u32,
                         outDim0: u32, outDim1: u32, outDim2: u32, outDim3: u32) -> u32 {
    // Compute output coordinates
    var remaining = outIdx;
    let out3 = remaining % outDim3;
    remaining = remaining / outDim3;
    let out2 = remaining % outDim2;
    remaining = remaining / outDim2;
    let out1 = remaining % outDim1;
    let out0 = remaining / outDim1;

    // Map to input coordinates (handle broadcasting)
    let in0 = select(out0, 0u, inDim0 == 1u);
    let in1 = select(out1, 0u, inDim1 == 1u);
    let in2 = select(out2, 0u, inDim2 == 1u);
    let in3 = select(out3, 0u, inDim3 == 1u);

    return in0 * inDim1 * inDim2 * inDim3 +
           in1 * inDim2 * inDim3 +
           in2 * inDim3 +
           in3;
}

@compute @workgroup_size(256, 1, 1)
fn broadcastAdd(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outIdx = global_id.x;
    if (outIdx >= broadcastParams.outputSize) {
        return;
    }

    let idxA = computeBroadcastIndex(outIdx,
                                     broadcastParams.dimA0, broadcastParams.dimA1,
                                     broadcastParams.dimA2, broadcastParams.dimA3,
                                     broadcastParams.dimOut0, broadcastParams.dimOut1,
                                     broadcastParams.dimOut2, broadcastParams.dimOut3);

    let idxB = computeBroadcastIndex(outIdx,
                                     broadcastParams.dimB0, broadcastParams.dimB1,
                                     broadcastParams.dimB2, broadcastParams.dimB3,
                                     broadcastParams.dimOut0, broadcastParams.dimOut1,
                                     broadcastParams.dimOut2, broadcastParams.dimOut3);

    broadcastOutput[outIdx] = broadcastA[idxA] + broadcastB[idxB];
}

@compute @workgroup_size(256, 1, 1)
fn broadcastMul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outIdx = global_id.x;
    if (outIdx >= broadcastParams.outputSize) {
        return;
    }

    let idxA = computeBroadcastIndex(outIdx,
                                     broadcastParams.dimA0, broadcastParams.dimA1,
                                     broadcastParams.dimA2, broadcastParams.dimA3,
                                     broadcastParams.dimOut0, broadcastParams.dimOut1,
                                     broadcastParams.dimOut2, broadcastParams.dimOut3);

    let idxB = computeBroadcastIndex(outIdx,
                                     broadcastParams.dimB0, broadcastParams.dimB1,
                                     broadcastParams.dimB2, broadcastParams.dimB3,
                                     broadcastParams.dimOut0, broadcastParams.dimOut1,
                                     broadcastParams.dimOut2, broadcastParams.dimOut3);

    broadcastOutput[outIdx] = broadcastA[idxA] * broadcastB[idxB];
}

// Fill with constant value
@compute @workgroup_size(256, 1, 1)
fn fill(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = scalarParams.alpha;
}

// Copy operation
@compute @workgroup_size(256, 1, 1)
fn copy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scalarParams.size) {
        return;
    }
    scalarOutput[idx] = scalarInput[idx];
}
