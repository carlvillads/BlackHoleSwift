//
//  SimulationData.swift
//  BlackHole
//
//  Created by Carl Villads Priisholm on 30/11/2025.
//  Based on the simulation by: https://github.com/kavan010/black_hole
//

import Foundation
import simd

struct CameraUniforms {
    var position: SIMD3<Float>; var _pad1: Float = 0
    var right: SIMD3<Float>; var _pad2: Float = 0
    var up: SIMD3<Float>; var _pad3: Float = 0
    var forward: SIMD3<Float>; var _pad4: Float = 0
    var tanHalfFov: Float
    var aspect: Float
    var moving: Bool
    var _pad5: Int32 = 0
}

struct DiskUniforms {
    var r1: Float
    var r2: Float
    var num: Float
    var thickness: Float
}

struct ObjectData {
    var posRadius: SIMD4<Float>
    var color: SIMD4<Float>
    var mass: Float
    var _padding: SIMD3<Float> = .zero
}

struct ObjectUniforms {
    var numObjects: Int32
}

struct FrameUniforms {
    var jitter: SIMD2<Float>
    var blendFactor: Float
    var _pad: Float = 0
}
