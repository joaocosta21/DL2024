import torch

# Define the file path to your attention weights
file_path = 'attn.pt'  # Replace with the correct path to your file

# Load the file
attn_weights = torch.load(file_path, map_location='cpu')  # Map to CPU to avoid memory issues

# Check the structure of the file and print the keys
if isinstance(attn_weights, dict):
    print("Keys in the file:", attn_weights.keys())
    
    # Assuming the heads are stored under a key like 'heads', modify as needed
    if 'heads' in attn_weights:
        heads = attn_weights['heads']
        print("Heads shape:", heads.shape if hasattr(heads, 'shape') else "Not a tensor")
        print("Heads data (preview):", heads)
    else:
        print("No 'heads' key found in the file.")
else:
    print("File does not contain a dictionary. It is of type:", type(attn_weights))
