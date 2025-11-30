//
//  MetalView.swift
//  BlackHole
//
//  Created by Carl Villads Priisholm on 30/11/2025.
//  Based on the simulation by: https://github.com/kavan010/black_hole
//

import SwiftUI
import MetalKit

struct MetalView: NSViewRepresentable {
    func makeCoordinator() -> Renderer {
        let device = MTLCreateSystemDefaultDevice()
        let metalView = MTKView(frame: .zero, device: device)
        return Renderer(metalView: metalView)!
    }
    
    func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView()
        mtkView.device = MTLCreateSystemDefaultDevice()
        mtkView.delegate = context.coordinator
        mtkView.framebufferOnly = false
        
        mtkView.layer?.contentsScale = 1.0  // retina is 4x upscaled by default
        return mtkView
    }
    
    func updateNSView(_ uiView: MTKView, context: Context) {}
}

struct ContentView: View {
    var body: some View {
        MetalView()
            .ignoresSafeArea()
    }
}

