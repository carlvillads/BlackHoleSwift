//
//  Renderer.swift
//  BlackHole
//
//  Created by Carl Villads Priisholm on 30/11/2025.
//  Based on the simulation by: https://github.com/kavan010/black_hole
//

import Foundation
import MetalKit

class Renderer: NSObject, MTKViewDelegate {
    var device: MTLDevice
    var commandQueue: MTLCommandQueue
    var computePipelineState: MTLComputePipelineState
    
    var cameraBuffer: MTLBuffer!
    var diskBuffer: MTLBuffer!
    var objectsBuffer: MTLBuffer!
    var objUniformsBuffer: MTLBuffer!
    var accumulationTexture: MTLTexture?
    var frameUniformsBuffer: MTLBuffer!
    
    var cameraPos: SIMD3<Float> = [0, 0, 6.34194e10]  // initial radius
    var target: SIMD3<Float> = [0, 0, 0]
    var time: Float = 0
    
    init?(metalView: MTKView) {
        self.device = metalView.device!
        self.commandQueue = device.makeCommandQueue()!
        
        do {
            let library = device.makeDefaultLibrary()!
            let kernel = library.makeFunction(name: "blackHoleCompute")!
            self.computePipelineState = try device.makeComputePipelineState(function: kernel)
        } catch {
            print("Shader compilation error: \(error)")
            return nil
        }
        
        super.init()
        self.setupBuffers()
    }
    
    func setupBuffers() {
        let r_s: Float = 1.269e10
        var disk = DiskUniforms(r1: r_s * 2.2, r2: r_s * 6.0, num: 2.0, thickness: 1e9)
        diskBuffer = device.makeBuffer(bytes: &disk, length: MemoryLayout<DiskUniforms>.stride, options: .storageModeShared)
        
        let objects: [ObjectData] = [
            ObjectData(posRadius: [4e11, 2e10, 0, 8e10], color: [1, 0.8, 0, 1], mass: 1.0, _padding: .zero),
            ObjectData(posRadius: [-4e11, -2e10, 2e11, 8e10], color: [1, 0, 0, 1], mass: 1.0, _padding: .zero)
        ]
        objectsBuffer = device.makeBuffer(bytes: objects, length: MemoryLayout<ObjectData>.stride * objects.count, options: .storageModeShared)
        
        var objUni = ObjectUniforms(numObjects: Int32(objects.count))
        objUniformsBuffer = device.makeBuffer(bytes: &objUni, length: MemoryLayout<ObjectUniforms>.stride, options: .storageModeShared)
        
        var frameUni = FrameUniforms(jitter: [0, 0], blendFactor: 0.5)
        frameUniformsBuffer = device.makeBuffer(bytes: &frameUni, length: MemoryLayout<FrameUniforms>.stride, options: .storageModeShared)
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    
    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        if accumulationTexture == nil ||
           accumulationTexture?.width != drawable.texture.width ||
           accumulationTexture?.height != drawable.texture.height {
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: drawable.texture.pixelFormat,
                width: drawable.texture.width,
                height: drawable.texture.height,
                mipmapped: false
            )
            desc.usage = [.shaderRead, .shaderWrite]
            accumulationTexture = device.makeTexture(descriptor: desc)
        }
        
        time += 0.005
        let radius: Float = 6.34194e10 * 2.5
        cameraPos = [radius * sin(time), radius * cos(time) * 0.3, radius * cos(time)]
        updateCameraBuffer(view)
        
        let jx = Float.random(in: -0.5...0.5)
        let jy = Float.random(in: -0.5...0.5)
        var frameData = FrameUniforms(jitter: [jx, jy], blendFactor: 0.65)
        memcpy(frameUniformsBuffer.contents(), &frameData, MemoryLayout<FrameUniforms>.stride)
        
        encoder.setComputePipelineState(computePipelineState)
        encoder.setTexture(drawable.texture, index: 0)
        encoder.setTexture(accumulationTexture, index: 1)
        
        encoder.setBuffer(cameraBuffer, offset: 0, index: 0)
        encoder.setBuffer(diskBuffer, offset: 0, index: 1)
        encoder.setBuffer(objUniformsBuffer, offset: 0, index: 2)
        encoder.setBuffer(objectsBuffer, offset: 0, index: 3)
        encoder.setBuffer(frameUniformsBuffer, offset: 0, index: 4)
        
        // threads
        let threadGroupSize = MTLSizeMake(16, 16, 1)
        let textureWidth = drawable.texture.width
        let textureHeight = drawable.texture.height
        
        let threadGroups = MTLSizeMake(
            (textureWidth + threadGroupSize.width - 1) / threadGroupSize.width,
            (textureHeight + threadGroupSize.height - 1) / threadGroupSize.height,
            1
        )
        
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
    
    func updateCameraBuffer(_ view: MTKView) {
        let fwd = normalize(target - cameraPos)
        let worldUp = SIMD3<Float>(0, 1, 0)
        let right = normalize(cross(fwd, worldUp))
        let up = cross(right, fwd)
        
        let aspect = Float(view.drawableSize.width / view.drawableSize.height)
        let tanHalfFov = tan(Float(70.0 * .pi / 180.0) * 0.5)
        
        var camData = CameraUniforms(
            position: cameraPos, _pad1: 0,
            right: right, _pad2: 0,
            up: up, _pad3: 0,
            forward: fwd, _pad4: 0,
            tanHalfFov: tanHalfFov,
            aspect: aspect,
            moving: false,
            _pad5: 0
        )
        
        if cameraBuffer == nil {
            cameraBuffer = device.makeBuffer(length: MemoryLayout<CameraUniforms>.stride, options: .storageModeShared)
        }
        memcpy(cameraBuffer.contents(), &camData, MemoryLayout<CameraUniforms>.stride)
    }
}
