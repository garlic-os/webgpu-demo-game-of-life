const GRID_SIZE = 32;
const UPDATE_INTERVAL = 100;
const WORKGROUP_SIZE = 8;
let step = 0;


const canvas = document.querySelector("canvas");
if (!canvas) throw new Error("No <canvas> element found");
// canvas.width = window.innerWidth;
// canvas.height = window.innerHeight;
canvas.width = 512;
canvas.height = 512;


if (!navigator.gpu) {
	throw new Error("Failed to access the WebGPU API");
}

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
	throw new Error("No supported GPU adapter found");
}

const device = await adapter.requestDevice();
const context = canvas.getContext("webgpu");
if (!context) throw new Error("Failed to create GPU canvas context");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
	device: device,
	format: canvasFormat,
});


const cellShaderModule = device.createShaderModule({
	label: "Cell shader",
	code: /* wgsl */`
		struct VertexInput {
			@location(0) pos: vec2f,
			@builtin(instance_index) instance: u32,
		};

		struct VertexOutput {
			@builtin(position) pos: vec4f,
			@location(0) cell: vec2f,
		};

		@group(0) @binding(0) var<uniform> grid: vec2f;
		@group(0) @binding(1) var<storage> cellState: array<u32>;


		// x, y, z, w
		@vertex
		fn vertexMain(input: VertexInput) -> VertexOutput {
			let i = f32(input.instance);
			let cell = vec2f(i % grid.x, floor(i / grid.x));
			let state = f32(cellState[input.instance]);

			let cellOffset = cell / grid * 2;
			let gridPos = (state * input.pos + 1) / grid - 1 + cellOffset;
			
			var output: VertexOutput;
			output.pos = vec4f(gridPos, 0, 1);
			output.cell = cell;
			return output;
		}


		// r, g, b, a
		@fragment
		fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
			let cellNormalized = input.cell / grid;
			return vec4f(cellNormalized, 1 - cellNormalized.x, 1);
		}
	`
});


const simulationShaderModule = device.createShaderModule({
	label: "Game of Life simulation shader",
	code: /* wgsl */ `
		@group(0) @binding(0) var<uniform> grid: vec2f;

		@group(0) @binding(1) var<storage> cellStateIn: array<u32>;
		@group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

		fn cellIndex(cell: vec2u) -> u32 {
			return (cell.y % u32(grid.y)) * u32(grid.x) +
			       (cell.x % u32(grid.x));
		}

		fn isCellActive(x: u32, y: u32) -> u32 {
			return cellStateIn[cellIndex(vec2(x, y))];
		}

		@compute
		@workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
		fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
			let numActiveNeighbors = isCellActive(cell.x+1, cell.y+1) +
			                         isCellActive(cell.x+1, cell.y  ) +
			                         isCellActive(cell.x+1, cell.y-1) +
			                         isCellActive(cell.x,   cell.y-1) +
			                         isCellActive(cell.x-1, cell.y-1) +
			                         isCellActive(cell.x-1, cell.y  ) +
			                         isCellActive(cell.x-1, cell.y+1) +
			                         isCellActive(cell.x,   cell.y+1);

			let i = cellIndex(cell.xy);

			// Game of Life rules
			switch numActiveNeighbors {
				case 2: {  // Active cells with 2 neighbors stay active
					cellStateOut[i] = cellStateIn[i];
				}
				case 3: {  // Cells with 3 neighbors become/stay active
					cellStateOut[i] = 1;
				}
				default: {  // Cells with neighbors fewer than 2 or more than 3
					cellStateOut[i] = 0;  // // become inactive
				}
			}
		}
	`
});


// A single square
const vertices = new Float32Array([
//     X     Y
	-0.8, -0.8,
	 0.8, -0.8,
	 0.8,  0.8,

	-0.8, -0.8,
	 0.8,  0.8,
	-0.8,  0.8,
]);

const vertexBuffer = device.createBuffer({
	label: "Call vertices",
	size: vertices.byteLength,
	usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(vertexBuffer, /*offset=*/0, vertices);

const vertexBufferLayout = {
	arrayStride: 8,
	attributes: [
		{
			format: "float32x2",
			offset: 0,
			shaderLocation: 0,
		},
	],
};

const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
const uniformBuffer = device.createBuffer({
	label: "Grid uniforms",
	size: uniformArray.byteLength,
	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

const cellStateArray = new Uint32Array(GRID_SIZE * GRID_SIZE);
const cellStateStorage = [
	device.createBuffer({
		label: "Cell state A",
		size: cellStateArray.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	}),
	device.createBuffer({
		label: "Cell state B",
		size: cellStateArray.byteLength,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	})
];

for (let i = 0; i < cellStateArray.length; i++) {
	cellStateArray[i] = Math.random() > 0.6 ? 1 : 0;
}
device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArray);


const bindGroupLayout = device.createBindGroupLayout({
	label: "Cell bind group layout",
	entries: [
		{
			binding: 0,
			visibility: GPUShaderStage.VERTEX |
			            GPUShaderStage.FRAGMENT |
			            GPUShaderStage.COMPUTE,
			buffer: {}  // Grid uniform buffer
		},
		{
			binding: 1,
			visibility: GPUShaderStage.VERTEX |
			            GPUShaderStage.FRAGMENT |
			            GPUShaderStage.COMPUTE,
			buffer: { type: "read-only-storage" }  // Cell state input buffer
		},
		{
			binding: 2,
			visibility: GPUShaderStage.COMPUTE,
			buffer: { type: "storage" }  // Cell state output buffer
		}
	]
});

const bindGroups = [
	device.createBindGroup({
		label: "Cell renderer bind group A",
		layout: bindGroupLayout,
		entries: [
			{
				binding: 0,
				resource: { buffer: uniformBuffer },
			},
			{
				binding: 1,
				resource: { buffer: cellStateStorage[0] },
			},
			{
				binding: 2,
				resource: { buffer: cellStateStorage[1] },
			},
		],
	}),
	device.createBindGroup({
		label: "Cell renderer bind group B",
		layout: bindGroupLayout,
		entries: [
			{
				binding: 0,
				resource: { buffer: uniformBuffer },
			},
			{
				binding: 1,
				resource: { buffer: cellStateStorage[1] },
			},
			{
				binding: 2,
				resource: { buffer: cellStateStorage[0] },
			},
		],
	}),
];


const pipelineLayout = device.createPipelineLayout({
	label: "Cell pipeline layout",
	bindGroupLayouts: [bindGroupLayout],
});

const cellPipeline = device.createRenderPipeline({
	label: "Cell pipeline",
	layout: pipelineLayout,
	vertex: {
		module: cellShaderModule,
		entryPoint: "vertexMain",
		buffers: [vertexBufferLayout]
	},
	fragment: {
		module: cellShaderModule,
		entryPoint: "fragmentMain",
		targets: [{
			format: canvasFormat
		}]
	},
});

const simulationPipeline = device.createComputePipeline({
	label: "Simulation pipeline",
	layout: pipelineLayout,
	compute: {
		module: simulationShaderModule,
		entryPoint: "computeMain",
	},
});


function updateGrid() {
	// if (!document.hasFocus()) return;
	const encoder = device.createCommandEncoder();

	{
		const pass = encoder.beginComputePass();
	
		pass.setPipeline(simulationPipeline);
		pass.setBindGroup(0, bindGroups[step % 2]);
	
		const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
		pass.dispatchWorkgroups(workgroupCount, workgroupCount);
	
		pass.end();
	}

	++step;
	
	{
		const pass = encoder.beginRenderPass({
			colorAttachments: [
				{
					view: context.getCurrentTexture().createView(),
					loadOp: "clear",
					clearValue: [ 0, 0, 0.4, 1 ],
					storeOp: "store",
				}
			]
		});
	
		pass.setPipeline(cellPipeline);
		pass.setBindGroup(0, bindGroups[step % 2]);
		pass.setVertexBuffer(0, vertexBuffer);
		pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE);
	
		pass.end();
	}

	const commandBuffer = encoder.finish();
	device.queue.submit([commandBuffer]);
}

setInterval(updateGrid, UPDATE_INTERVAL);
