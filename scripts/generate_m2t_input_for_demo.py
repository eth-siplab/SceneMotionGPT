input_filename = '/home/dominik/Documents/repos/MotionGPT/datasets/humanml3d/test.txt'
output_filename = 'demos/demo_test_m2t.txt'
base_path = '/home/dominik/Documents/repos/MotionGPT/datasets/humanml3d/new_joint_vecs'
template = 'What is being shown in <motion>? Please describe it in text.#'

# --- Script ---
with open(input_filename, 'r') as f_in, open(output_filename, 'w') as f_out:
    for file_id in f_in:
        # Clean up the line (removes newline characters) and build the full path
        full_path = f"{base_path}/{file_id.strip()}.npy"
        
        # Write the complete line to the output file
        f_out.write(f"{template}{full_path}\n")

print(f"Successfully generated '{output_filename}' with {sum(1 for line in open(input_filename))} lines.")