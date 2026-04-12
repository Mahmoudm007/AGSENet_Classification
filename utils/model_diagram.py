import os

def generate_mermaid_diagram(output_path):
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>AGSENet Model Architecture</title>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({ startOnLoad: true, theme: 'default', maxTextSize: 10000 });
    </script>
    <style>
        body { font-family: sans-serif; text-align: center; background-color: #f9f9f9; padding: 20px; }
        .mermaid { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); display: inline-block; overflow: auto; max-width: 95%; }
    </style>
</head>
<body>
    <h1>AGSENet Classification Architecture</h1>
    <p>Interactive Diagram. Hover or zoom to inspect the details.</p>
    <div class="mermaid">
    graph TD
        classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
        classDef encoder fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
        classDef csi fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
        classDef pool fill:#fffde7,stroke:#fbc02d,stroke-width:1px,stroke-dasharray: 5 5;
        classDef decoder fill:#fff3e0,stroke:#e65100,stroke-width:2px;
        classDef ssie fill:#ffebee,stroke:#c62828,stroke-width:2px;
        classDef classification fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
        
        Input([Image Input HxWx3]):::input --> EN1[Encoder S1: RSU7]:::encoder
        EN1 --> CSIF1[CSIF 1]:::csi
        CSIF1 --> Pool1((MaxPool)):::pool
        Pool1 --> EN2[Encoder S2: RSU6]:::encoder
        EN2 --> CSIF2[CSIF 2]:::csi
        CSIF2 --> Pool2((MaxPool)):::pool
        Pool2 --> EN3[Encoder S3: RSU5]:::encoder
        EN3 --> CSIF3[CSIF 3]:::csi
        CSIF3 --> Pool3((MaxPool)):::pool
        Pool3 --> EN4[Encoder S4: RSU4]:::encoder
        EN4 --> CSIF4[CSIF 4]:::csi
        CSIF4 --> Pool4((MaxPool)):::pool
        Pool4 --> EN5[Encoder S5: RSU4F]:::encoder
        EN5 --> CSIF5[CSIF 5]:::csi
        CSIF5 --> Pool5((MaxPool)):::pool
        Pool5 --> EN6[Encoder S6: RSU4F]:::encoder
        EN6 --> CSIF6[CSIF 6]:::csi
        
        %% Top Down Fusion
        CSIF6 --> Proj6[Proj 6: ConvBNReLU]:::decoder
        CSIF5 --> Proj5[Proj 5: ConvBNReLU]:::decoder
        CSIF4 --> Proj4[Proj 4: ConvBNReLU]:::decoder
        CSIF3 --> Proj3[Proj 3: ConvBNReLU]:::decoder

        Proj6 --> SSIE5[SSIE Fusion 5]:::ssie
        Proj5 --> SSIE5
        
        SSIE5 --> SSIE4[SSIE Fusion 4]:::ssie
        Proj4 --> SSIE4
        
        SSIE4 --> SSIE3[SSIE Fusion 3]:::ssie
        Proj3 --> SSIE3
        
        %% Feature aggregation
        SSIE3 -->|Adaptive Avg/Max Pool| GAP3((Gap/Max 3)):::pool
        SSIE4 -->|Adaptive Avg/Max Pool| GAP4((Gap/Max 4)):::pool
        SSIE5 -->|Adaptive Avg/Max Pool| GAP5((Gap/Max 5)):::pool
        Proj6 -->|Adaptive Avg/Max Pool| GAP6((Gap/Max 6)):::pool
        
        GAP3 --> Concat[Flatten & Concat]:::classification
        GAP4 --> Concat
        GAP5 --> Concat
        GAP6 --> Concat
        
        Concat --> FC1[Linear -> ReLU -> Dropout]:::classification
        FC1 --> FC2[Linear Out]:::classification
        FC2 --> Output([Softmax Predictions]):::classification
    </div>
</body>
</html>
"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(html_content)
        
    print(f"Model diagram successfully saved to {output_path}")

if __name__ == "__main__":
    generate_mermaid_diagram('outputs/model_architecture_diagram.html')
