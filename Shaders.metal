//
//  Shaders.metal
//  BlackHole
//
//  Created by Carl Villads Priisholm on 30/11/2025.
//  Based on the simulation by: https://github.com/kavan010/black_hole
//

#include <metal_stdlib>
using namespace metal;

constant float SagA_rs = 1.269e10f;
constant float D_LAMBDA = 2e8f;
constant float ESCAPE_R = 1.0e13f;

struct CameraUniforms {
    float3 position;
    float _pad1;
    float3 right;
    float _pad2;
    float3 up;
    float _pad3;
    float3 forward;
    float _pad4;
    float tanHalfFov;
    float aspect;
    bool moving;
    int _pad5;
};

struct DiskUniforms {
    float r1;
    float r2;
    float num;
    float thickness;
};

struct ObjectData {
    float4 posRadius; // xyz = pos, w = radius
    float4 color;
    float mass;
    float3 _padding;
};

struct ObjectUniforms {
    int numObjects;
};

struct RayState {
    float r, theta, phi;
    float dr, dtheta, dphi;
};

struct FrameUniforms {
    float2 jitter;
    float blendFactor;
    float _pad;
};

void schwarzchildDerivatives(RayState state, float E, thread float3 &dPos, thread float3 &dVel) {
    float r = state.r;
    float s_theta = sin(state.theta);
    float c_theta = cos(state.theta);
    
    if (abs(s_theta) < 1e-4f) s_theta = 1e-4f;
    
    // schwarzchild factor
    float f = 1.0f - SagA_rs / r;
    float dt_dL = E / f;
    
    dPos = float3(state.dr, state.dtheta, state.dphi);
    
    // geodesic equations
    float term1    = (SagA_rs / (2.0f * r * r * f)) * state.dr * state.dr;
    float term2    = r * f * (state.dtheta * state.dtheta + s_theta * s_theta * state.dphi * state.dphi);
    float termTime = -(SagA_rs * f) / (2.0f * r * r) * dt_dL * dt_dL;
    
    dVel.x = termTime + term1 + term2;                                                                      // d/dL(dr)
    dVel.y = (-2.0f / r) * state.dr * state.dtheta + s_theta * c_theta * state.dphi * state.dphi;           // d/dL(dtheta)
    dVel.z = (-2.0f / r) * state.dr * state.dphi - 2.0f * (c_theta / s_theta) * state.dtheta * state.dphi;  // d/dL(dphi)
}

void RK4Step(thread RayState &s, float E, float h) {
    float3 dp1, dv1, dp2, dv2, dp3, dv3, dp4, dv4;
    
    schwarzchildDerivatives(s, E, dp1, dv1);  // k1
    
    RayState s2 = s;
    s2.r      += dp1.x * h * 0.5f;
    s2.theta  += dp1.y * h * 0.5f;
    s2.phi    += dp1.z * h * 0.5f;
    s2.dr     += dv1.x * h * 0.5f;
    s2.dtheta += dv1.y * h * 0.5f;
    s2.dphi   += dv1.z * h * 0.5f;
    schwarzchildDerivatives(s2, E, dp2, dv2);  // k2
    
    RayState s3 = s;
    s3.r      += dp2.x * h * 0.5f;
    s3.theta  += dp2.y * h * 0.5f;
    s3.phi    += dp2.z * h * 0.5f;
    s3.dr     += dv2.x * h * 0.5f;
    s3.dtheta += dv2.y * h * 0.5f;
    s3.dphi   += dv2.z * h * 0.5f;
    schwarzchildDerivatives(s3, E, dp3, dv3);  // k3
    
    RayState s4 = s;
    s4.r      += dp3.x * h * 0.5f;
    s4.theta  += dp3.y * h * 0.5f;
    s4.phi    += dp3.z * h * 0.5f;
    s4.dr     += dv3.x * h * 0.5f;
    s4.dtheta += dv3.y * h * 0.5f;
    s4.dphi   += dv3.z * h * 0.5f;
    schwarzchildDerivatives(s4, E, dp4, dv4);  // k4
    
    // average
    s.r      += (h / 0.5f) * (dp1.x + 2*dp2.x + 2*dp3.x + dp4.x);
    s.theta  += (h / 0.5f) * (dp1.y + 2*dp2.y + 2*dp3.y + dp4.y);
    s.phi    += (h / 0.5f) * (dp1.z + 2*dp2.z + 2*dp3.z + dp4.z);
    s.dr     += (h / 0.5f) * (dv1.x + 2*dv2.x + 2*dv3.x + dv4.x);
    s.dtheta += (h / 0.5f) * (dv1.y + 2*dv2.y + 2*dv3.y + dv4.y);
    s.dphi   += (h / 0.5f) * (dv1.z + 2*dv2.z + 2*dv3.z + dv4.z);
}

kernel void blackHoleCompute(
                             texture2d<float, access::write> output [[texture(0)]],
                             texture2d<float, access::read_write> history [[texture(1)]],
                             constant CameraUniforms &cam [[buffer(0)]],
                             constant DiskUniforms &disk [[buffer(1)]],
                             constant ObjectUniforms &objUniforms [[buffer(2)]],
                             constant ObjectData *objects [[buffer(3)]],
                             constant FrameUniforms &frame [[buffer(4)]],
                             uint2 gid [[thread_position_in_grid]]
                             ) {
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
    
    // Ray generation
    float u = (2.0f * (float(gid.x) + 0.5f + frame.jitter.x) / float(output.get_width()) - 1.0f) * cam.aspect * cam.tanHalfFov;
    float v = (1.0f - 2.0f * (float(gid.y) + 0.5f + frame.jitter.y) / float(output.get_height())) * cam.tanHalfFov;
    float3 dir = normalize(u * cam.right - v * cam.up + cam.forward);
    float3 pos = cam.position;
    
    // Init state
    RayState ray;
    ray.r = length(pos);
    ray.theta = acos(clamp(pos.z / (ray.r + 1e-5f), -1.0f, 1.0f));
    ray.phi = atan2(pos.y, pos.x);
    
    float st = sin(ray.theta); if (abs(st) < 1e-4f) st = 1e-4f;
    float ct = cos(ray.theta);
    float sp = sin(ray.phi);
    float cp = cos(ray.phi);
    
    ray.dr     = st * cp * dir.x + st * sp * dir.y + ct * dir.z;
    ray.dtheta = (ct * cp * dir.x + ct * sp * dir.y - st * dir.z) / ray.r;
    ray.dphi   = (-sp * dir.x + cp * dir.y) / (ray.r * st);
    
    float f    = 1.0f - SagA_rs / ray.r;
    float term = (ray.dr * ray.dr) / f + ray.r * ray.r * (ray.dtheta * ray.dtheta + st * st * ray.dphi * ray.dphi);
    float E    = f * sqrt(max(term, 0.0f));
    
    float3 prevPosCartesan = pos;
    
    // accumulation loop
    float3 accumColor = float3(0.0f);
    float transmittance = 1.0f;
    int steps = 250;
    
    for (int i = 0; i < steps; ++i) {
        // Horizon Block
        if (ray.r <= SagA_rs * 1.01f) {
            transmittance = 0.0f;
            break;
        }
        
        // Escape
        if (ray.r > ESCAPE_R) break;
        
        float distRatio = (ray.r - SagA_rs) / SagA_rs;
        float stepMultiplier = clamp(distRatio, 0.1f, 8.0f);
        RK4Step(ray, E, D_LAMBDA * stepMultiplier);
        
        // Cartesian conversion
        st = sin(ray.theta); if (abs(st) < 1e-4f) st = 1e-4f;
        float3 newPos;
        newPos.x = ray.r * st * cos(ray.phi);
        newPos.y = ray.r * st * sin(ray.phi);
        newPos.z = ray.r * cos(ray.theta);
        
        // Disk Intersection
        if (prevPosCartesan.y * newPos.y < 0.0f) {
            float t = prevPosCartesan.y / (prevPosCartesan.y - newPos.y);
            float3 hitPoint = prevPosCartesan + (newPos - prevPosCartesan) * t;
            float r_disk = length(hitPoint.xz);
    
            if (r_disk >= disk.r1 && r_disk <= disk.r2) {
                float intensity = (r_disk - disk.r1) / (disk.r2 - disk.r1);
                float noise = sin(atan2(hitPoint.x, hitPoint.z) * 10.0f) * 0.1f;
                intensity = exp(-3.0 * abs(intensity - 0.5 + noise));
                
                float3 diskColor = float3(1.0f, 0.6f * intensity, 0.2f * intensity) * intensity * 2.0f;
                float alpha = 0.4f * intensity;
                
                accumColor += diskColor * alpha * transmittance;
                transmittance *= (1.0f - alpha);
            }
        }
        
        // Object Intersection
        for (int j = 0; j < objUniforms.numObjects; j++) {
            float3 center = objects[j].posRadius.xyz;
            float rad = objects[j].posRadius.w;
            float3 diff = newPos - center;
            if (dot(diff, diff) <= rad * rad) {
                accumColor += objects[j].color.rgb * transmittance;
                transmittance = 0.0f;
                break;
            }
        }
        
        if (transmittance <= 0.01f) break;
        prevPosCartesan = newPos;
    }
    
    float4 finalColor = float4(accumColor, 1.0f);
    float4 prevColor = history.read(gid);
    float4 blendedColor = mix(finalColor, prevColor, frame.blendFactor);
    
    output.write(blendedColor, gid);
    history.write(blendedColor, gid);
}
