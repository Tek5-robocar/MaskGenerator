Shader "Custom/RandomDynamicShapes"
{
    Properties
    {
        _MaxDensity("Max Density", Range(0, 1)) = 0.5
        _SizeVariation("Size Variation", Range(0, 1)) = 0.5
        _AnimationSpeed("Animation Speed", Float) = 1.0
        _ShapeVariety("Shape Variety", Range(0, 1)) = 0.8
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            float _MaxDensity;
            float _SizeVariation;
            float _AnimationSpeed;
            float _ShapeVariety;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            // Hash function for pseudo-random numbers
            float hash(float2 p)
            {
                float3 p3 = frac(float3(p.xyx) * 0.13);
                p3 += dot(p3, p3.yzx + 3.33);
                return frac((p3.x + p3.y) * p3.z);
            }

            // Rotate function
            float2 rotate(float2 p, float angle)
            {
                float s = sin(angle);
                float c = cos(angle);
                return float2(p.x * c - p.y * s, p.x * s + p.y * c);
            }

            // Circle shape
            float drawCircle(float2 p, float size)
            {
                return smoothstep(size, size * 0.9, length(p));
            }

            // Square shape
            float drawSquare(float2 p, float size)
            {
                float2 d = abs(p) - size;
                return smoothstep(0.01, 0.0, max(d.x, d.y));
            }

            // Triangle shape (renamed from 'triangle')
            float drawTriangle(float2 p, float size)
            {
                float k = sqrt(3.0);
                p.x = abs(p.x) - size;
                p.y = p.y + size/k;
                if(p.x + k*p.y > 0.0) p = float2(p.x - k*p.y, -k*p.x - p.y)/2.0;
                p.x -= clamp(p.x, -2.0*size, 0.0);
                return smoothstep(0.01, 0.0, -length(p)*sign(p.y));
            }

            // Cross shape
            float drawCross(float2 p, float size)
            {
                float2 d = abs(p) - float2(size, size*0.3);
                float cross1 = smoothstep(0.01, 0.0, max(d.x, d.y));
                float cross2 = smoothstep(0.01, 0.0, max(d.y, d.x));
                return max(cross1, cross2);
            }

            float4 frag (v2f i) : SV_Target
            {
                float2 uv = i.uv;
                float time = _Time.y * _AnimationSpeed;
                
                // Generate random density for this frame
                float density = frac(hash(float2(time, 0.123)) * 1.618) * _MaxDensity;
                
                // Scale factor to adjust the grid based on density
                float scale = lerp(50.0, 5.0, density);
                
                // Grid coordinates
                float2 gridPos = uv * scale;
                float2 cellPos = floor(gridPos);
                float2 localPos = frac(gridPos) - 0.5;
                
                // Initialize color
                float4 col = float4(0, 0, 0, 1);
                
                // Check neighboring cells for better coverage
                for (int y = -1; y <= 1; y++)
                {
                    for (int x = -1; x <= 1; x++)
                    {
                        float2 neighbor = float2(x, y);
                        float2 cell = cellPos + neighbor;
                        
                        // Random seed for this cell
                        float seed = hash(cell + float2(time, 0.0));
                        
                        // Only create a shape in this cell based on density
                        if (seed > density) continue;
                        
                        // Random position within cell
                        float2 offset = float2(hash(cell + float2(1.0, 0.0)), hash(cell + float2(0.0, 1.0))) - 0.5;
                        float2 shapePos = localPos - neighbor - offset;
                        
                        // Random rotation
                        float angle = hash(cell + float2(2.0, 0.0)) * 6.283;
                        shapePos = rotate(shapePos, angle);
                        
                        // Random size
                        float size = 0.1 + hash(cell + float2(3.0, 0.0)) * _SizeVariation;
                        
                        // Random shape selection based on variety
                        float shapeSelector = hash(cell + float2(4.0, 0.0));
                        float shape = 0.0;
                        
                        if (shapeSelector < _ShapeVariety * 0.25)
                            shape = drawCircle(shapePos, size);
                        else if (shapeSelector < _ShapeVariety * 0.5)
                            shape = drawSquare(shapePos, size);
                        else if (shapeSelector < _ShapeVariety * 0.75)
                            shape = drawTriangle(shapePos, size);
                        else
                            shape = drawCross(shapePos, size);
                        
                        // Add to color
                        col.rgb += shape;
                    }
                }
                
                // Clamp the color
                col.rgb = saturate(col.rgb);
                return col;
            }
            ENDCG
        }
    }
}