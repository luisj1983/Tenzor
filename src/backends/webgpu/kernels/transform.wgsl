// Tensor transformation operations: transpose, reshape, permute, etc.

struct TransposeParams {
    dim0: u32,
    dim1: u32,
    dim2: u32,
    dim3: u32,
    perm0: u32,
    perm1: u32,
    perm2: u32,
    perm3: u32,
    ndim: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: TransposeParams;

// General N-D transpose
@compute @workgroup_size(256, 1, 1)
fn transpose(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    let dims = array<u32, 4>(params.dim0, params.dim1, params.dim2, params.dim3);
    let perm = array<u32, 4>(params.perm0, params.perm1, params.perm2, params.perm3);

    let totalSize = params.dim0 * params.dim1 * params.dim2 * params.dim3;

    if (idx >= totalSize) {
        return;
    }

    // Compute input indices
    var inIndices: array<u32, 4>;
    var remaining = idx;

    for (var i = 0u; i < params.ndim; i = i + 1u) {
        let dimSize = dims[params.ndim - 1u - i];
        inIndices[params.ndim - 1u - i] = remaining % dimSize;
        remaining = remaining / dimSize;
    }

    // Permute indices
    var outIndices: array<u32, 4>;
    for (var i = 0u; i < params.ndim; i = i + 1u) {
        outIndices[i] = inIndices[perm[i]];
    }

    // Compute output linear index
    var outIdx = 0u;
    var stride = 1u;

    for (var i = 0u; i < params.ndim; i = i + 1u) {
        let dim = params.ndim - 1u - i;
        outIdx = outIdx + outIndices[dim] * stride;
        stride = stride * dims[perm[dim]];
    }

    output[outIdx] = input[idx];
}

// Optimized 2D transpose (matrix transpose)
struct Transpose2DParams {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<storage, read> input2d: array<f32>;
@group(0) @binding(1) var<storage, read_write> output2d: array<f32>;
@group(0) @binding(2) var<uniform> params2d: Transpose2DParams;

const TILE_DIM: u32 = 16u;

var<workgroup> tile: array<f32, 256>; // TILE_DIM * TILE_DIM

@compute @workgroup_size(16, 16, 1)
fn transpose2d(@builtin(global_invocation_id) global_id: vec3<u32>,
               @builtin(local_invocation_id) local_id: vec3<u32>,
               @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    let row = workgroup_id.y * TILE_DIM + local_id.y;
    let col = workgroup_id.x * TILE_DIM + local_id.x;

    // Load tile into shared memory
    if (row < params2d.rows && col < params2d.cols) {
        tile[local_id.y * TILE_DIM + local_id.x] = input2d[row * params2d.cols + col];
    }

    workgroupBarrier();

    // Write transposed tile
    let outRow = workgroup_id.x * TILE_DIM + local_id.y;
    let outCol = workgroup_id.y * TILE_DIM + local_id.x;

    if (outRow < params2d.cols && outCol < params2d.rows) {
        output2d[outRow * params2d.rows + outCol] = tile[local_id.x * TILE_DIM + local_id.y];
    }
}

// Flatten operation
struct FlattenParams {
    totalSize: u32,
}

@group(0) @binding(0) var<storage, read> flattenInput: array<f32>;
@group(0) @binding(1) var<storage, read_write> flattenOutput: array<f32>;
@group(0) @binding(2) var<uniform> flattenParams: FlattenParams;

@compute @workgroup_size(256, 1, 1)
fn flatten(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= flattenParams.totalSize) {
        return;
    }
    flattenOutput[idx] = flattenInput[idx];
}

// Reshape (just a memory copy for contiguous tensors)
@compute @workgroup_size(256, 1, 1)
fn reshape(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= flattenParams.totalSize) {
        return;
    }
    flattenOutput[idx] = flattenInput[idx];
}

// Squeeze - remove dimensions of size 1
@compute @workgroup_size(256, 1, 1)
fn squeeze(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= flattenParams.totalSize) {
        return;
    }
    flattenOutput[idx] = flattenInput[idx];
}

// Unsqueeze - add dimensions of size 1
@compute @workgroup_size(256, 1, 1)
fn unsqueeze(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= flattenParams.totalSize) {
        return;
    }
    flattenOutput[idx] = flattenInput[idx];
}

// Slice operation
struct SliceParams {
    inputDim0: u32,
    inputDim1: u32,
    inputDim2: u32,
    inputDim3: u32,
    outputDim0: u32,
    outputDim1: u32,
    outputDim2: u32,
    outputDim3: u32,
    start0: u32,
    start1: u32,
    start2: u32,
    start3: u32,
    step0: u32,
    step1: u32,
    step2: u32,
    step3: u32,
}

@group(0) @binding(0) var<storage, read> sliceInput: array<f32>;
@group(0) @binding(1) var<storage, read_write> sliceOutput: array<f32>;
@group(0) @binding(2) var<uniform> sliceParams: SliceParams;

@compute @workgroup_size(256, 1, 1)
fn slice(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outIdx = global_id.x;

    let totalOutputSize = sliceParams.outputDim0 * sliceParams.outputDim1 *
                         sliceParams.outputDim2 * sliceParams.outputDim3;

    if (outIdx >= totalOutputSize) {
        return;
    }

    // Compute output coordinates
    var remaining = outIdx;
    let out3 = remaining % sliceParams.outputDim3;
    remaining = remaining / sliceParams.outputDim3;
    let out2 = remaining % sliceParams.outputDim2;
    remaining = remaining / sliceParams.outputDim2;
    let out1 = remaining % sliceParams.outputDim1;
    let out0 = remaining / sliceParams.outputDim1;

    // Map to input coordinates
    let in0 = sliceParams.start0 + out0 * sliceParams.step0;
    let in1 = sliceParams.start1 + out1 * sliceParams.step1;
    let in2 = sliceParams.start2 + out2 * sliceParams.step2;
    let in3 = sliceParams.start3 + out3 * sliceParams.step3;

    // Compute input index
    let inIdx = in0 * sliceParams.inputDim1 * sliceParams.inputDim2 * sliceParams.inputDim3 +
                in1 * sliceParams.inputDim2 * sliceParams.inputDim3 +
                in2 * sliceParams.inputDim3 +
                in3;

    sliceOutput[outIdx] = sliceInput[inIdx];
}

// Concatenation along a dimension
struct ConcatParams {
    dim: u32,
    numTensors: u32,
    outputSize: u32,
    outputDim0: u32,
    outputDim1: u32,
    outputDim2: u32,
    outputDim3: u32,
}

@group(0) @binding(0) var<storage, read> concatInput0: array<f32>;
@group(0) @binding(1) var<storage, read> concatInput1: array<f32>;
@group(0) @binding(2) var<storage, read_write> concatOutput: array<f32>;
@group(0) @binding(3) var<uniform> concatParams: ConcatParams;

// Note: This is simplified for 2 inputs. For arbitrary number, use dynamic dispatch
@compute @workgroup_size(256, 1, 1)
fn concat(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= concatParams.outputSize) {
        return;
    }

    // This is a simplified version - actual implementation would need
    // to handle arbitrary number of inputs and split points
    concatOutput[idx] = select(concatInput0[idx], concatInput1[idx], idx >= concatParams.outputSize / 2u);
}

// Stack tensors along a new dimension
@compute @workgroup_size(256, 1, 1)
fn stack(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= concatParams.outputSize) {
        return;
    }

    let tensorSize = concatParams.outputSize / concatParams.numTensors;
    let tensorIdx = idx / tensorSize;
    let elemIdx = idx % tensorSize;

    // Select appropriate input based on tensor index
    concatOutput[idx] = select(concatInput0[elemIdx], concatInput1[elemIdx], tensorIdx == 1u);
}

// Split tensor along a dimension
struct SplitParams {
    inputSize: u32,
    splitSize: u32,
    numSplits: u32,
    splitIdx: u32,
}

@group(0) @binding(0) var<storage, read> splitInput: array<f32>;
@group(0) @binding(1) var<storage, read_write> splitOutput: array<f32>;
@group(0) @binding(2) var<uniform> splitParams: SplitParams;

@compute @workgroup_size(256, 1, 1)
fn split(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outIdx = global_id.x;

    if (outIdx >= splitParams.splitSize) {
        return;
    }

    let inIdx = splitParams.splitIdx * splitParams.splitSize + outIdx;
    splitOutput[outIdx] = splitInput[inIdx];
}

// Tile/repeat tensor
struct TileParams {
    inputDim0: u32,
    inputDim1: u32,
    inputDim2: u32,
    inputDim3: u32,
    repeat0: u32,
    repeat1: u32,
    repeat2: u32,
    repeat3: u32,
}

@group(0) @binding(0) var<storage, read> tileInput: array<f32>;
@group(0) @binding(1) var<storage, read_write> tileOutput: array<f32>;
@group(0) @binding(2) var<uniform> tileParams: TileParams;

@compute @workgroup_size(256, 1, 1)
fn tile(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outIdx = global_id.x;

    let outputDim0 = tileParams.inputDim0 * tileParams.repeat0;
    let outputDim1 = tileParams.inputDim1 * tileParams.repeat1;
    let outputDim2 = tileParams.inputDim2 * tileParams.repeat2;
    let outputDim3 = tileParams.inputDim3 * tileParams.repeat3;

    let totalOutputSize = outputDim0 * outputDim1 * outputDim2 * outputDim3;

    if (outIdx >= totalOutputSize) {
        return;
    }

    // Compute output coordinates
    var remaining = outIdx;
    let out3 = remaining % outputDim3;
    remaining = remaining / outputDim3;
    let out2 = remaining % outputDim2;
    remaining = remaining / outputDim2;
    let out1 = remaining % outputDim1;
    let out0 = remaining / outputDim1;

    // Map to input coordinates (modulo)
    let in0 = out0 % tileParams.inputDim0;
    let in1 = out1 % tileParams.inputDim1;
    let in2 = out2 % tileParams.inputDim2;
    let in3 = out3 % tileParams.inputDim3;

    // Compute input index
    let inIdx = in0 * tileParams.inputDim1 * tileParams.inputDim2 * tileParams.inputDim3 +
                in1 * tileParams.inputDim2 * tileParams.inputDim3 +
                in2 * tileParams.inputDim3 +
                in3;

    tileOutput[outIdx] = tileInput[inIdx];
}

// Flip/reverse along dimension
struct FlipParams {
    dim0: u32,
    dim1: u32,
    dim2: u32,
    dim3: u32,
    flipDim: u32,
    totalSize: u32,
}

@group(0) @binding(0) var<storage, read> flipInput: array<f32>;
@group(0) @binding(1) var<storage, read_write> flipOutput: array<f32>;
@group(0) @binding(2) var<uniform> flipParams: FlipParams;

@compute @workgroup_size(256, 1, 1)
fn flip(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= flipParams.totalSize) {
        return;
    }

    let dims = array<u32, 4>(flipParams.dim0, flipParams.dim1, flipParams.dim2, flipParams.dim3);

    // Compute coordinates
    var coords: array<u32, 4>;
    var remaining = idx;

    for (var i = 0u; i < 4u; i = i + 1u) {
        coords[3u - i] = remaining % dims[3u - i];
        remaining = remaining / dims[3u - i];
    }

    // Flip the specified dimension
    coords[flipParams.flipDim] = dims[flipParams.flipDim] - 1u - coords[flipParams.flipDim];

    // Compute output index
    let outIdx = coords[0] * dims[1] * dims[2] * dims[3] +
                 coords[1] * dims[2] * dims[3] +
                 coords[2] * dims[3] +
                 coords[3];

    flipOutput[outIdx] = flipInput[idx];
}
