// Indexing and gathering operations

struct GatherParams {
    inputDim0: u32,
    inputDim1: u32,
    inputDim2: u32,
    inputDim3: u32,
    indexCount: u32,
    axis: u32,
    outputSize: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: GatherParams;

// Gather operation along specified axis
@compute @workgroup_size(256, 1, 1)
fn gather(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outIdx = global_id.x;

    if (outIdx >= params.outputSize) {
        return;
    }

    let dims = array<u32, 4>(params.inputDim0, params.inputDim1, params.inputDim2, params.inputDim3);

    // Compute output coordinates
    var coords: array<u32, 4>;
    var remaining = outIdx;

    for (var i = 0u; i < 4u; i = i + 1u) {
        let dim = 3u - i;
        coords[dim] = remaining % dims[dim];
        remaining = remaining / dims[dim];
    }

    // Replace coordinate at gather axis with index value
    let indexPos = outIdx / (params.outputSize / params.indexCount);
    coords[params.axis] = indices[indexPos];

    // Compute input linear index
    let inIdx = coords[0] * dims[1] * dims[2] * dims[3] +
                coords[1] * dims[2] * dims[3] +
                coords[2] * dims[3] +
                coords[3];

    output[outIdx] = input[inIdx];
}

// Scatter operation
struct ScatterParams {
    outputDim0: u32,
    outputDim1: u32,
    outputDim2: u32,
    outputDim3: u32,
    indexCount: u32,
    axis: u32,
    inputSize: u32,
}

@group(0) @binding(0) var<storage, read> scatterInput: array<f32>;
@group(0) @binding(1) var<storage, read> scatterIndices: array<u32>;
@group(0) @binding(2) var<storage, read_write> scatterOutput: array<f32>;
@group(0) @binding(3) var<uniform> scatterParams: ScatterParams;

@compute @workgroup_size(256, 1, 1)
fn scatter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let inIdx = global_id.x;

    if (inIdx >= scatterParams.inputSize) {
        return;
    }

    let dims = array<u32, 4>(
        scatterParams.outputDim0, scatterParams.outputDim1,
        scatterParams.outputDim2, scatterParams.outputDim3
    );

    // Compute input coordinates
    var coords: array<u32, 4>;
    var remaining = inIdx;

    for (var i = 0u; i < 4u; i = i + 1u) {
        let dim = 3u - i;
        coords[dim] = remaining % dims[dim];
        remaining = remaining / dims[dim];
    }

    // Get scatter index
    let indexPos = inIdx / (scatterParams.inputSize / scatterParams.indexCount);
    coords[scatterParams.axis] = scatterIndices[indexPos];

    // Compute output linear index
    let outIdx = coords[0] * dims[1] * dims[2] * dims[3] +
                 coords[1] * dims[2] * dims[3] +
                 coords[2] * dims[3] +
                 coords[3];

    scatterOutput[outIdx] = scatterInput[inIdx];
}

// Index select - gather along dimension 0
struct IndexSelectParams {
    inputSize: u32,
    outputSize: u32,
    stride: u32,
    indexCount: u32,
}

@group(0) @binding(0) var<storage, read> selectInput: array<f32>;
@group(0) @binding(1) var<storage, read> selectIndices: array<u32>;
@group(0) @binding(2) var<storage, read_write> selectOutput: array<f32>;
@group(0) @binding(3) var<uniform> selectParams: IndexSelectParams;

@compute @workgroup_size(256, 1, 1)
fn indexSelect(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outIdx = global_id.x;

    if (outIdx >= selectParams.outputSize) {
        return;
    }

    let batchIdx = outIdx / selectParams.stride;
    let offset = outIdx % selectParams.stride;

    let inputIdx = selectIndices[batchIdx];
    let inIdx = inputIdx * selectParams.stride + offset;

    selectOutput[outIdx] = selectInput[inIdx];
}

// Masked fill - fill values where mask is true
struct MaskedFillParams {
    size: u32,
    fillValue: f32,
}

@group(0) @binding(0) var<storage, read> maskInput: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<u32>;  // 0 or 1
@group(0) @binding(2) var<storage, read_write> maskOutput: array<f32>;
@group(0) @binding(3) var<uniform> maskParams: MaskedFillParams;

@compute @workgroup_size(256, 1, 1)
fn maskedFill(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= maskParams.size) {
        return;
    }

    maskOutput[idx] = select(maskInput[idx], maskParams.fillValue, mask[idx] != 0u);
}

// Masked select - select values where mask is true
@group(0) @binding(0) var<storage, read> maskedSelectInput: array<f32>;
@group(0) @binding(1) var<storage, read> selectMask: array<u32>;
@group(0) @binding(2) var<storage, read_write> maskedSelectOutput: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputCount: array<u32>;

struct MaskedSelectParams {
    size: u32,
}

@group(0) @binding(4) var<uniform> maskedSelectParams: MaskedSelectParams;

var<workgroup> sharedCount: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn maskedSelect(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>) {
    let idx = global_id.x;
    let tid = local_id.x;

    var localCount = 0u;

    // Count selected elements
    if (idx < maskedSelectParams.size) {
        localCount = selectMask[idx];
    }

    sharedCount[tid] = localCount;
    workgroupBarrier();

    // Prefix sum to find output position
    for (var offset = 1u; offset < 256u; offset = offset * 2u) {
        var temp = 0u;
        if (tid >= offset) {
            temp = sharedCount[tid - offset];
        }
        workgroupBarrier();

        if (tid >= offset) {
            sharedCount[tid] = sharedCount[tid] + temp;
        }
        workgroupBarrier();
    }

    // Write output
    if (idx < maskedSelectParams.size && selectMask[idx] != 0u) {
        let outPos = sharedCount[tid] - 1u;
        maskedSelectOutput[outPos] = maskedSelectInput[idx];
    }

    // Last thread updates total count
    if (tid == 255u) {
        atomicAdd(&outputCount[0], sharedCount[255]);
    }
}

// Take - advanced indexing with multi-dimensional indices
struct TakeParams {
    inputSize: u32,
    indexCount: u32,
}

@group(0) @binding(0) var<storage, read> takeInput: array<f32>;
@group(0) @binding(1) var<storage, read> takeIndices: array<u32>;
@group(0) @binding(2) var<storage, read_write> takeOutput: array<f32>;
@group(0) @binding(3) var<uniform> takeParams: TakeParams;

@compute @workgroup_size(256, 1, 1)
fn take(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= takeParams.indexCount) {
        return;
    }

    let inputIdx = takeIndices[idx];

    if (inputIdx < takeParams.inputSize) {
        takeOutput[idx] = takeInput[inputIdx];
    } else {
        takeOutput[idx] = 0.0;  // Out of bounds
    }
}

// Embedding lookup
struct EmbeddingParams {
    numEmbeddings: u32,
    embeddingDim: u32,
    indexCount: u32,
}

@group(0) @binding(0) var<storage, read> embeddingTable: array<f32>;
@group(0) @binding(1) var<storage, read> embeddingIndices: array<u32>;
@group(0) @binding(2) var<storage, read_write> embeddingOutput: array<f32>;
@group(0) @binding(3) var<uniform> embeddingParams: EmbeddingParams;

@compute @workgroup_size(256, 1, 1)
fn embedding(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outIdx = global_id.x;

    let totalOutputSize = embeddingParams.indexCount * embeddingParams.embeddingDim;

    if (outIdx >= totalOutputSize) {
        return;
    }

    let indexPos = outIdx / embeddingParams.embeddingDim;
    let dimPos = outIdx % embeddingParams.embeddingDim;

    let embeddingIdx = embeddingIndices[indexPos];
    let tableIdx = embeddingIdx * embeddingParams.embeddingDim + dimPos;

    embeddingOutput[outIdx] = embeddingTable[tableIdx];
}

// One-hot encoding
struct OneHotParams {
    indexCount: u32,
    numClasses: u32,
}

@group(0) @binding(0) var<storage, read> onehotIndices: array<u32>;
@group(0) @binding(1) var<storage, read_write> onehotOutput: array<f32>;
@group(0) @binding(2) var<uniform> onehotParams: OneHotParams;

@compute @workgroup_size(256, 1, 1)
fn oneHot(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outIdx = global_id.x;

    let totalOutputSize = onehotParams.indexCount * onehotParams.numClasses;

    if (outIdx >= totalOutputSize) {
        return;
    }

    let indexPos = outIdx / onehotParams.numClasses;
    let classPos = outIdx % onehotParams.numClasses;

    let targetClass = onehotIndices[indexPos];

    onehotOutput[outIdx] = select(0.0, 1.0, classPos == targetClass);
}

// Advanced indexing with boolean mask
struct BooleanMaskParams {
    inputDim0: u32,
    inputDim1: u32,
    inputDim2: u32,
    inputDim3: u32,
    outputSize: u32,
}

@group(0) @binding(0) var<storage, read> boolInput: array<f32>;
@group(0) @binding(1) var<storage, read> boolMask: array<u32>;
@group(0) @binding(2) var<storage, read_write> boolOutput: array<f32>;
@group(0) @binding(3) var<uniform> boolParams: BooleanMaskParams;

@compute @workgroup_size(256, 1, 1)
fn booleanMask(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    let totalSize = boolParams.inputDim0 * boolParams.inputDim1 *
                   boolParams.inputDim2 * boolParams.inputDim3;

    if (idx >= totalSize) {
        return;
    }

    // This is simplified - actual implementation would need
    // prefix sum to compute output positions
    if (boolMask[idx] != 0u) {
        // Would need atomic counter or prefix sum
        boolOutput[idx] = boolInput[idx];
    }
}

// Put - inverse of take
@compute @workgroup_size(256, 1, 1)
fn put(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= takeParams.indexCount) {
        return;
    }

    let outputIdx = takeIndices[idx];

    if (outputIdx < takeParams.inputSize) {
        takeOutput[outputIdx] = takeInput[idx];
    }
}
